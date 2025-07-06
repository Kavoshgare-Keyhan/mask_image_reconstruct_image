import numpy as np
import pandas as pd
import argparse, math, sys, os, random
import torch
from torch import nn
import distributed as dist
from transformers import DistilBertForMaskedLM, DistilBertConfig
from vqvae import FlatVQVAE
import neptune.new as neptune

from torchvision.models import resnet50, ResNet50_Weights
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.optim as optim

run = neptune.init_run(
    project="tns/Vqvae-transformer",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwODg4OTU0Yy0xODAyLTRiM2QtYjYzYi0xMWQxYThmYWJlOWQifQ==",
    capture_stdout = False,
    capture_stderr = False,
    # with_id="MAS-389"
)

def full_mask(quantizes, indices, n_token, mask_perc=1):
    mask_pattern = np.random.default_rng().choice([True, False], size=(1,1, n_token), p=[mask_perc, 0])
    mask_quantizes = quantizes.clone() # shallow copy
    mask_quantizes[mask_pattern] = 0  # Assuming 0 is the mask token
    mask_indices = indices.clone()
    mask_indices[~mask_pattern[0]] = -100 # Assuming -100 is the mask label token
    return mask_quantizes, mask_indices, mask_pattern

def custom_mask(quantizes, indices, n_token, mask_pattern, unmask_pattern):
    if unmask_pattern: mask_pattern[0,0,unmask_pattern] = False
    mask_quantizes = quantizes.clone() # shallow copy
    mask_quantizes[mask_pattern] = 0  # Assuming 0 is the mask token
    mask_indices = indices.clone()
    mask_indices[~mask_pattern[0]] = -100 # Assuming -100 is the mask label token
    return mask_quantizes, mask_indices, mask_pattern

def random_mask(quantizes, indices, n_token, mask_perc, mask_pattern=None):
    if mask_pattern is None:
        mask_pattern = np.random.default_rng().choice([True, False], size=(1,1, n_token), p=[mask_perc, 1 - mask_perc])
    else:
        pass # plan to update mask pattern rather than replace it completely
    mask_quantizes = quantizes.clone()
    mask_quantizes[mask_pattern] = 0  # Assuming 0 is the mask token
    mask_indices = indices.clone()
    mask_indices[~mask_pattern[0]] = -100 # Assuming -100 is the mask label token
    return mask_quantizes, mask_indices, mask_pattern[0][0]

def selective_masking(distil, quantized, indices, mask_percentage):
    total_num = quantized.shape[1] 
    total_unmasked_number = (int) (total_num * (1-mask_percentage))
    unmask_index = (int) (total_num/2)
    quantized_masked = torch.zeros_like(quantized)
    mask_pattern = torch.ones(quantized.shape[:2], dtype=torch.bool)
    already_unmasked = set()
    for i in range(0,total_unmasked_number):
        mask_pattern[0,unmask_index] = False
        already_unmasked.add(unmask_index)
        quantized_masked[0, unmask_index] = quantized[0,unmask_index]
        outputs = distil(inputs_embeds = quantized_masked, output_hidden_states = True)
        max_logits, max_indices = torch.max(outputs.logits, dim=-1)
        sorted_logits, sorted_indices = torch.sort(max_logits[0])
        for min_index in sorted_indices:
            if min_index.item() not in already_unmasked:
                unmask_index = min_index.item()
                break
    indices_masked = indices.clone()
    indices_masked[~mask_pattern[0]] = -100
    return quantized_masked, indices_masked, mask_pattern[0]

def selective_mask_update(distil, quantized, indices, d_embed_vec, n_token=400):
    mask_pattern = torch.zeros(quantized.shape[:2])
    outputs = distil(inputs_embeds = quantized_masked, output_hidden_states = True)
    max_logits, max_indices = torch.max(outputs.logits, dim=-1)
    sorted_logits, sorted_indices = torch.sort(max_logits[0])
    sort_np = sorted_indices.cpu().numpy()
    return sort_np


def load_embedding_space(path='/home/abghamtm/work/masking_comparison/checkpoint/vqvae/quantized_epoch80_flat_vqvae80x80_144x456codebook.npy'):
    quantizes = np.load(path)
    quant_b = quantizes
    n, c, h, w = quantizes.shape
    quantizes = quantizes.transpose(0, 2, 3, 1)
    quantizes = quantizes.reshape(n, h * w, c)
    return quantizes, quant_b

def load_indices(path='/home/abghamtm/work/masking_comparison/checkpoint/vqvae/indices_epoch80_flat_vqvae80x80_144x456codebook.npy'):
    indices = np.load(path)
    n, h, w = indices.shape
    indices = indices.reshape(n, h * w)
    return indices

def load_labels(path='/home/abghamtm/work/masking_comparison/checkpoint/vqvae/labels_epoch80_flat_vqvae80x80_144x456codebook.npy'):
    labels = np.load(path)
    # labels = torch.from_numpy(labels)
    return labels

def vqvae_model_setup(ckpt_vqvae, device):
    model_vqvae = FlatVQVAE().to(device)
    model_vqvae.load_state_dict(torch.load(ckpt_vqvae, map_location=device))
    model_vqvae = model_vqvae.to(device)
    model_vqvae.eval()
    return model_vqvae
    
def setup_resources():
    torch.cuda.set_device(3)
    torch.cuda.empty_cache()
    # if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    #     device = [f"cuda:{i}" for i in range(torch.cuda.device_count())] # need to becustomized to select not running GPUs
    # else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device

def transformer_setup(device, args, n_token, vocab_size, d_embed_vec):
    cfg = DistilBertConfig(
            vocab_size = vocab_size,
            hidden_size = d_embed_vec,
            sinusoidal_pos_embds = False,
            n_layers = 6,
            n_heads = 4,
            max_position_embeddings = n_token
    )
    model_distil = DistilBertForMaskedLM(cfg).to(device)
    model_distil.load_state_dict(torch.load(args.ckpt_distil_combined))
    model_distil = model_distil.to(device)
    model_distil.eval()
    return model_distil

def vqvae_setup(device, args):
    model_vqvae = FlatVQVAE().to(device)
    model_vqvae.load_state_dict(torch.load(args.ckpt_vqvae, map_location=device))
    model_vqvae = model_vqvae.to(device)
    model_vqvae.eval()
    return model_vqvae

def mask_iter(min, max, step, n_token, arbitrary_percentage:list =[.85,.95]):
    mask_percentages = np.arange(min, max, step)
    mask_percentages = np.append(mask_percentages, arbitrary_percentage) if arbitrary_percentage else mask_percentages
    mask_percentages = np.sort(mask_percentages)
    reverse_mask_percentages = mask_percentages[::-1] # reverse the sorted array
    mask_perc_map_indices_length = n_token * reverse_mask_percentages
    return mask_percentages, reverse_mask_percentages, mask_perc_map_indices_length

def main(args):
    # sample_list = [462,1671,1836,4970,5852,7777,8513,8 685, 9469, 9644] # This list is also drawn from one of the trial of the generated random samples
    device = setup_resources()
    quantizes, quant_b = load_embedding_space()
    indices = load_indices()
    labels = load_labels()

    n_sample = quantizes.shape[0]
    d_embed_vec = quantizes.shape[2]
    n_token = np.prod(quantizes.shape[1])
    quantizes = quantizes.reshape((n_sample, n_token, d_embed_vec))
    length = int(math.sqrt(n_token))
    indices = indices.reshape((n_sample, n_token))
    indices_to_sort = set(indices.flatten())
    indices_to_sort = sorted(indices_to_sort)
    vocab_size = indices_to_sort[-1] + 1

    model_distil = transformer_setup(device, args, n_token, vocab_size, d_embed_vec)
    model_vqvae = vqvae_setup(device, args)

    mask_percentages, reverse_mask_percentages, mask_perc_map_indices_length = mask_iter(min=0.1, max=1.1, step=0.1, n_token=n_token)

    print(n_sample)
    n_sample = 10_000
    reconstruction_error = np.zeros((n_sample, len(reverse_mask_percentages)))
    criterion = nn.MSELoss()
    for x in range(n_sample): 
        q = torch.from_numpy(quantizes[x]).to(device)
        q = torch.reshape(q, (1, q.size(dim=0), q.size(dim=1)))
        index = torch.from_numpy(indices[x]).to(device)
        label = labels[x]

        '''
        Outer increments: 12
        Inner increments: 4+4+4+4+8+8+8+8+8+8+8 = 72
        '''
        n_iter = 0
        for i in range(len(mask_perc_map_indices_length)):
            n_iter += 1  # for the outer loop
            if i < len(mask_perc_map_indices_length) - 1:
                n_iter += int((mask_perc_map_indices_length[i] - mask_perc_map_indices_length[i+1]) // 5)

        stack_mask_pattern = np.zeros((n_iter, n_token), dtype=bool)
        stack_output_logits = np.zeros((n_iter, n_token, vocab_size), dtype=float) # 80 is the number of events, 400 is the number of tokens represents each pixel of encoded latent space which has the shape of 20*20, and 456 is the number of codebook vectors
        unmask_indices = []
        i = 0 # iteration on mask_perc_map_indices_length
        l = 0 # iteration on the number of total iterations corresponding to the number of mask and logits generation (number of calling the transformer model)

        q_masked, index_masked, mask_pattern = full_mask(q, index, n_token)   
        q_masked = q_masked.to(device)
        index_masked = index_masked.to(device)
        index_masked_for_visual = index.clone()

        while i < len(mask_perc_map_indices_length):
            print('additive approach - event:', x)           
            with torch.no_grad():
                outputs = model_distil(inputs_embeds = q_masked, output_hidden_states = True)
                stack_output_logits[l] = outputs.logits[0].detach().cpu().numpy() # save the logits for each iteration
                stack_mask_pattern[l] = mask_pattern[0][0] # save the mask pattern for each iteration
                k += 1
                confidence, ind_most_probable = torch.max(outputs.logits, dim=2) # the first variable is the max_logits, and the second variable is the indices of the max logits
                confidence_based_recons_index = ind_most_probable[0]
                confidence_based_recons_index = confidence_based_recons_index.to(device)
                # Flatten the tensor to 1D
                confidence_flat = confidence.flatten()
                # Get indices that would sort the tensor in ascending order
                conf_indices_min_max = torch.argsort(confidence_flat, dim=0)
            # stack_mask_pattern[i] = mask_pattern[0][0]

            if unmask_indices:
                confidence_based_recons_index[unmask_indices] = index[unmask_indices]
                
            with torch.no_grad():
                distil_out = model_vqvae.decode_code(torch.reshape(confidence_based_recons_index, (1, length,length)).to(device))

                vqvae_out = model_vqvae.decode(torch.from_numpy(quant_b[x]).to(device)) #torch.reshape(torch.from_numpy(indices[x]), (1,length,length)).to(device)
                
                index_masked_for_visual[mask_pattern[0][0]] = 0
                vqvae_masked_out = model_vqvae.decode_code(torch.reshape(index_masked_for_visual, (1,length,length)).to(device))

            percentage = int(reverse_mask_percentages[i]*100)
            img_list = [vqvae_out, vqvae_masked_out, distil_out]
            # label_list = [vqvae_img_label.item(), vqvae_masked_img_label.item(), add_mask_img_label.item()]
            for ii, img in enumerate(img_list):
                img_np = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
                if ii==0 and percentage==100:
                    np.save(f'/home/abghamtm/work/masking_comparison/image/recons/original_img_reconstrcted_data/{str(x).zfill(5)}_OriginalImage_TrueLabel={label}.npy', img_np)
                elif ii==1:
                    np.save(f'/home/abghamtm/work/masking_comparison/image/recons/aditive_img_data/{str(x).zfill(5)}_MaskPercentage={percentage}_AdditiveMasking_TrueLabel={label}.npy', img_np)
                else:
                    np.save(f'/home/abghamtm/work/masking_comparison/image/recons/aditive_img_data/{str(x).zfill(5)}_MaskPercentage={percentage}_AdditiveMaskingWithTransformer_TrueLabel={label}.npy', img_np)
            
            recons_loss = criterion(distil_out, vqvae_out.unsqueeze(0))
            reconstruction_error[x, i] = recons_loss.item()

            try: # the purpose of this try-except block is to avoid the error when the mask_perc_map_indices_length index i+1 reaches 12
                # unmask indices gradually
                for j in range(int((mask_perc_map_indices_length[i]-mask_perc_map_indices_length[i+1])//5)):
                    c = 0
                    for k in range(len(conf_indices_min_max)):
                        if conf_indices_min_max[k].item() not in unmask_indices:
                            unmask_indices.append(conf_indices_min_max[k].item())
                            c += 1
                            if c == 5:
                                break
                    
                    q_masked, index_masked, mask_pattern = custom_mask(q, index, n_token, mask_pattern, unmask_indices)
                    if (q_masked[unmask_indices] == 0).all(): # I have to divide num_zeros by q_masked.shape[2] because q_masked is a 3D tensor and we want to check the number of zeros per token/embedded vector
                        raise ValueError(f"Unmask codebooks are populated by zeros")                    
                   
                    outputs = model_distil(inputs_embeds = q_masked, output_hidden_states = True)
                    stack_output_logits[l] = outputs.logits[0].detach().cpu().numpy() # save the logits for each iteration
                    stack_mask_pattern[l] = mask_pattern[0][0] # save the mask pattern for each iteration
                    k += 1
                    confidence, ind_most_probable = torch.max(outputs.logits, dim=2)
                    confidence_flat = confidence.flatten()
                    conf_indices_min_max = torch.argsort(confidence_flat, dim=0)
            except:
                break

            i+=1
        np.save(f'/home/abghamtm/work/masking_comparison/masking-reconstruction_pattern/additiv_mask_pattern_{str(x).zfill(5)}.npy', stack_mask_pattern)

    reconstruction_err = np.mean(reconstruction_error, axis=0)
    reconstruction_err = reconstruction_err[::-1]
    # np.save('/home/abghamtm/work/masking_comparison/additive_masking_reconstruction_error.npy', reconstruction_error)
    run["recons/average_mse_additive_masking_reconstruction"].log(list(reconstruction_err))

    average_errors = []
    for perc in mask_percentages:
        reconstruction_errors = []
        stack_mask_pattern = []
        correct_random_pred = 0
        tot_sample = 0
        for x in range(n_sample):
            print('random approach - event:', x)
            q = torch.from_numpy(quantizes[x]).to(device)
            index = torch.from_numpy(indices[x]).to(device)
            q = torch.reshape(q, (1, q.size(dim=0), q.size(dim=1)))
            label = labels[x]
            
            with torch.no_grad():
                q_masked, index_masked, mask_pattern = random_mask(q, index , n_sample, n_token, perc)                                
                q_masked = q_masked.to(device)
                index_masked = index_masked.to(device)
                # must be stacked *********
                np.save(f'/home/abghamtm/work/masking_comparison/masking-reconstruction_pattern/random_mask_pattern_{x}_MaskPercentage={int(perc*100)}.npy', mask_pattern)

            #Fill in predicted tokens
            with torch.no_grad():
                outputs = model_distil(inputs_embeds = q_masked, output_hidden_states = True)
                # check q_masked zeros and if they do not match number of masked raise #*************************
                ind_most_probable = torch.argmax(outputs.logits, dim=2)
                confidence_based_recons_index = index
                for p in range(0,n_token):
                    if(mask_pattern[p]):
                        #confidence_based_recons_index[p] = ind_most_probable.detach().cpu().numpy()[0][p] 
                        confidence_based_recons_index[p] = ind_most_probable[0][p] 
                
                #Reconstruct with distil predictions
                confidence_based_recons_index = confidence_based_recons_index.to(device)
                distil_out = model_vqvae.decode_code(torch.reshape(confidence_based_recons_index, (1,length,length)).to(device))

                #Reconstruct Original
                vqvae_out = model_vqvae.decode(torch.from_numpy(quant_b[x]).to(device)) #torch.reshape(torch.from_numpy(indices[x]), (1,length,length)).to(device)
                index_masked_forvis = index.clone()
                index_masked_forvis[mask_pattern]=0
                vqvae_masked_out = model_vqvae.decode_code(torch.reshape(index_masked_forvis, (1,length,length)).to(device))

                percentage = int(perc*100)
                img_list = [vqvae_masked_out, distil_out]
                # label_list = [vqvae_masked_img_label.item(), rand_mask_img_label.item()]
                for ii, img in enumerate(img_list):
                    img_np = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    if ii==0:
                        np.save(f'/home/abghamtm/work/masking_comparison/image/recons/random_img_data/{str(x).zfill(5)}_MaskPercentage={percentage}_RandomMasking_TrueLabel={label}.npy', img_np)
                        # loaded_img = np.load(f'/home/abghamtm/work/masking_comparison/image/recons/comparison/raw_img_data/{str(x).zfill(5)}_MaskPercentage={percentage}_RandomMasking_label={label}.npy')
                    else:
                        np.save(f'/home/abghamtm/work/masking_comparison/image/recons/random_img_data/{str(x).zfill(5)}_MaskPercentage={percentage}_RandomMaskingWithTransformer_TrueLabel={label}.npy', img_np)

            recon_loss = criterion(distil_out, vqvae_out)
            reconstruction_errors.append(recon_loss.item())
        run["recons/average_mse_random_masking_reconstruction"].log(np.mean(reconstruction_errors))
        average_errors.append(np.mean(reconstruction_errors))

    selective_recons_avg_err = []
    for perc in mask_percentages:
        reconstruction_errors = []
        correct_grad_pred = 0
        tot_sample = 0
        for x in range(n_sample):
            print('selective approach - event:', x)
            q = torch.from_numpy(quantizes[x]).to(device)
            index = torch.from_numpy(indices[x]).to(device)
            q = torch.reshape(q, (1, q.size(dim=0), q.size(dim=1)))
            label = labels[x]
            
            with torch.no_grad():
                q_masked, index_masked, mask_pattern = selective_masking(model_distil, q, index ,perc)  
                # count number of zeros of confidence # **************************                            
                q_masked = q_masked.to(device)
                index_masked = index_masked.to(device)

            np.save(f'/home/abghamtm/work/masking_comparison/masking-reconstruction_pattern/selective_mask_pattern_{x}_MaskPercentage={int(perc*100)}.npy', mask_pattern)

            with torch.no_grad():
                outputs = model_distil(inputs_embeds = q_masked, output_hidden_states = True)
                ind_most_probable = torch.argmax(outputs.logits, dim=2)
                confidence_based_recons_index = index
                for p in range(0,n_token):
                    if(mask_pattern[p]):
                        confidence_based_recons_index[p] = ind_most_probable[0][p] 

                #Reconstruct with distil predictions
                confidence_based_recons_index = confidence_based_recons_index.to(device)
                distil_out = model_vqvae.decode_code(torch.reshape(confidence_based_recons_index, (1,length,length)).to(device))

                #Reconstruct Original
                vqvae_out = model_vqvae.decode(torch.from_numpy(quant_b[x]).to(device))
                index_masked_forvis = index.clone()
                index_masked_forvis[mask_pattern]=0
                vqvae_masked_out = model_vqvae.decode_code(torch.reshape(index_masked_forvis, (1,length,length)).to(device))

                percentage = int(perc*100)
                img_list = [vqvae_masked_out, distil_out]
                # label_list = [vqvae_masked_img_label.item(), grad_mask_img_label.item()]
                for ii, img in enumerate(img_list):
                    img_np = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    if ii==0:
                        np.save(f'/home/abghamtm/work/masking_comparison/image/recons/selective_img_data/{str(x).zfill(5)}_MaskPercentage={percentage}_SelectiveMasking_TrueLabel={label}.npy', img_np)
                    else:
                        np.save(f'/home/abghamtm/work/masking_comparison/image/recons/selective_img_data/{str(x).zfill(5)}_MaskPercentage={percentage}_SelectiveMaskingWithTransformer_TrueLabel={label}.npy', img_np)

            recon_loss = criterion(distil_out, vqvae_out)
            reconstruction_errors.append(recon_loss.item())
        run["recons/average_mse_selective_masking_reconstruction"].log(np.mean(reconstruction_errors))
        selective_recons_avg_err.append(np.mean(reconstruction_errors))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)

    port = (
        2 ** 15
        + 2 ** 14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )
    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument('--ckpt_vqvae', type=str, default="/home/abghamtm/work/masking_comparison/checkpoint/vqvae/model_epoch80_flat_vqvae80x80_144x456codebook.pth")
    parser.add_argument('--ckpt_distil_combined', type=str, default="/home/abghamtm/work/masking_comparison/checkpoint/distil/80x80_100ClassImagenet_flat_144x456codebook_75mask_epoch100.pt")


    args = parser.parse_args()
    dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))