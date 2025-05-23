import numpy as np
import pandas as pd
import argparse, math, sys, os, random
import torch
from torch import nn
import torchvision.utils as vutils
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import distributed as dist
from transformers import DistilBertForMaskedLM, DistilBertConfig
from vqvae import FlatVQVAE
from PIL import Image
import neptune.new as neptune
from torchvision.models import resnet50, ResNet50_Weights

run = neptune.init_run(
    project="tns/Vqvae-transformer",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwODg4OTU0Yy0xODAyLTRiM2QtYjYzYi0xMWQxYThmYWJlOWQifQ==",
    capture_stdout = False,
    capture_stderr = False,
    # with_id="MAS-389"
)
def additive_mask(quantizes, indices, n_token, unmask_pattern=None):
    mask_pattern = np.random.default_rng().choice([True, False], size=(1,1, n_token), p=[1, 0])
    if unmask_pattern: mask_pattern[0,0,unmask_pattern] = False
    mask_quantizes = quantizes.clone() # shallow copy
    mask_quantizes[mask_pattern] = 0  # Assuming 0 is the mask token
    mask_indices = indices.clone()
    mask_indices[~mask_pattern[0]] = -100 # Assuming -100 is the mask label token
    return mask_quantizes, mask_indices, mask_pattern[0][0]

def random_mask(quantizes, indices, n_sample, n_token, mask_perc):
    mask_pattern = np.random.default_rng().choice([True, False], size=(1,1, n_token), p=[mask_perc, 1 - mask_perc])
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
    return quantized_masked,indices_masked, mask_pattern[0]

def main(args):
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.distributed = dist.get_world_size() > 1

    indices = np.load('/home/abghamtm/work/masking_comparison/checkpoint/vqvae/indices_epoch80_flat_vqvae80x80_144x456codebook.npy')
    n, h, w = indices.shape
    indices = indices.reshape(n, h * w)

    quantizes = np.load('/home/abghamtm/work/masking_comparison/checkpoint/vqvae/quantized_epoch80_flat_vqvae80x80_144x456codebook.npy')
    quant_b = quantizes
    n, c, h, w = quantizes.shape
    quantizes = quantizes.transpose(0, 2, 3, 1)
    quantizes = quantizes.reshape(n, h * w, c)

    #Bottom data and parameters
    n_sample = quantizes.shape[0]
    d_embed_vec = quantizes.shape[2]
    n_token = np.prod(quantizes.shape[1])
    quantizes = quantizes.reshape((n_sample, n_token, d_embed_vec))
    length = int(math.sqrt(n_token))
    indices = indices.reshape((n_sample, n_token))
    indices_to_sort = set(indices.flatten())
    indices_to_sort = sorted(indices_to_sort)
    vocab_size = indices_to_sort[-1] + 1

    #Define Distilbert model
    cfg = DistilBertConfig(
            vocab_size=vocab_size,
            hidden_size=d_embed_vec,
            sinusoidal_pos_embds=False,
            n_layers=6,
            n_heads=4,
            max_position_embeddings=n_token
    )
    model_distil = DistilBertForMaskedLM(cfg).to(device)
    model_distil.load_state_dict(torch.load(args.ckpt_distil_combined))
    model_distil = model_distil.to(device)
    model_distil.eval()

    #Define VQVAE model
    model_vqvae = FlatVQVAE().to(device)
    model_vqvae.load_state_dict(torch.load(args.ckpt_vqvae, map_location=device))
    model_vqvae = model_vqvae.to(device)
    model_vqvae.eval()

    # Define classifier and load saved model(weights)
    weights = ResNet50_Weights.IMAGENET1K_V2
    preprocess = weights.transforms()
    classifier = resnet50(pretrained=False)
    classifier.load_state_dict(torch.load('/home/abghamtm/work/masking_comparison/checkpoint/classifier/resnet50/specialized_weights_epoch10.pth'))
    classifier.to(device)
    classifier.eval()

    mask_percentages = np.arange(0.1, 1.1, 0.1)
    mask_percentages = np.append(mask_percentages,[.85,.95])
    mask_percentages = np.sort(mask_percentages)
    reverse_mask_percentages = mask_percentages[::-1] # reverse the sorted array
    mask_perc_map_indices_length = n_token * reverse_mask_percentages
    # sample_list = np.random.choice(n_sample, 10, replace=False)
    sample_list = [462,1671,1836,4970,5852,7777,8513,8685,9469,9644] # This list is also drawn from one of the trial of the generated random samples

    reconstruction_error = np.zeros((n_sample,len(reverse_mask_percentages)))
    classification_acc = np.zeros((n_sample,len(reverse_mask_percentages)))

    criterion = nn.MSELoss()
    for x in sample_list: 
        print('additive approach - event:', x)
        q = torch.from_numpy(quantizes[x]).to(device)
        q = torch.reshape(q, (1, q.size(dim=0), q.size(dim=1)))
        index = torch.from_numpy(indices[x]).to(device)

        unmask_indices = []
        i = 0
        while i < len(mask_perc_map_indices_length):
            q_masked, index_masked, mask_pattern = additive_mask(q, index , n_token, unmask_indices)   
            q_masked = q_masked.to(device)
            index_masked = index_masked.to(device)
            print('masking step:', mask_perc_map_indices_length[i])
            print('number of tokens: ', n_token)
            print('mask percentage: ', (mask_perc_map_indices_length[i]/n_token)*100)
            print('quantize shape:', q_masked.shape)
            print('index shape:', index_masked.shape)
            print('mask shape:', mask_pattern.shape)
            with torch.no_grad():
                outputs = model_distil(inputs_embeds = q_masked, output_hidden_states = True)
                prediction_value, confidence_based_prediction = torch.max(outputs.logits, dim=2)
                confidence_based_recons_index = confidence_based_prediction[0]
                # print('indices should be replaced by original index:', unmask_indices)
                np.save(f'/home/abghamtm/work/masking_comparison/masking-reconstruction_pattern/additiv_mask_pattern_{x}_MaskingPerc={(mask_perc_map_indices_length[i]/n_token)*100}.npy', mask_pattern)
                if unmask_indices:
                    confidence_based_recons_index[unmask_indices] = index[unmask_indices]
                
                confidence_based_recons_index = confidence_based_recons_index.to(device)
                distil_out = model_vqvae.decode_code(torch.reshape(confidence_based_recons_index, (1,length,length)).to(device))

                vqvae_out = model_vqvae.decode(torch.from_numpy(quant_b[x]).to(device)) #torch.reshape(torch.from_numpy(indices[x]), (1,length,length)).to(device)
                index_masked_forvis = index.clone()
                index_masked_forvis[mask_pattern]=0
                vqvae_masked_out = model_vqvae.decode_code(torch.reshape(index_masked_forvis, (1,length,length)).to(device))
                print('original image shape after decoding:', vqvae_out.shape)
                print('Masked image w/o transformer shape after decoding:', vqvae_masked_out.shape)
                print('Masked image w transformer shape after decoding', distil_out.shape)

                # Label outputs
                vqvae_out = vqvae_out.unsqueeze(0)
                vqvae_img = preprocess(vqvae_out)
                vqvae_img = vqvae_img.to(device)
                vqvae_img_prob = classifier(vqvae_img)
                _, vqvae_img_label = torch.max(vqvae_img_prob, 1)

                # vqvae_masked_out = vqvae_masked_out.unsqueeze(0)
                vqvae_masked_img = preprocess(vqvae_masked_out)
                vqvae_masked_img = vqvae_masked_img.to(device)
                vqvae_masked_img_prob = classifier(vqvae_masked_img)
                _, vqvae_masked_img_label = torch.max(vqvae_masked_img_prob, 1)
                
                add_mask_img = preprocess(distil_out)
                add_mask_img = add_mask_img.to(device)
                add_mask_img_prob = classifier(add_mask_img)
                _, add_mask_img_label = torch.max(add_mask_img_prob, 1)

                percentage = int(reverse_mask_percentages[i]*100)
                img_list = [vqvae_out, vqvae_masked_out, distil_out]
                label_list = [vqvae_img_label.item(), vqvae_masked_img_label.item(), add_mask_img_label.item()]
                for ii, (img, label) in enumerate(zip(img_list, label_list)):
                    # Move tensor to CPU and convert to NumPy array
                    img_np = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
                    
                    plt.figure()
                    plt.imshow(img_np, cmap='viridis')
                    # plt.xlabel(label)
                    # plt.colorbar()
                    if ii==0 and percentage==100:
                        plt.savefig(f'image/recons/comparison/{str(x).zfill(5)}_Original_label={label}.png')
                    elif ii==1:
                        img_path1 = f'image/recons/comparison/{str(x).zfill(5)}_MaskPercentage={percentage}_AdditiveMasking_label={label}.png'
                        plt.savefig(img_path1)
                    else:
                        img_path2 = f'image/recons/comparison/{str(x).zfill(5)}_MaskPercentage={percentage}_AdditiveMaskingWithTransformer_label={label}.png'
                        plt.savefig(img_path2)
                    plt.close()

            recons_loss = criterion(distil_out, vqvae_out)
            reconstruction_error[x, i] = recons_loss.item()
            
            classification_acc[x, i] = (add_mask_img_label == vqvae_img_label).sum().item()

            try:
                for j in range(int((mask_perc_map_indices_length[i]-mask_perc_map_indices_length[i+1])//5)):
                    top5 = []
                    while len(top5)<5:
                        min_index = torch.argmin(prediction_value).item()
                        if min_index not in unmask_indices:
                            unmask_indices.append(min_index)
                            top5.append(min_index)
                        prediction_value[0,min_index] = prediction_value.max().item()
                    q_masked, index_masked, mask_pattern = additive_mask(q, index , n_token, unmask_indices) 
                    outputs = model_distil(inputs_embeds = q_masked, output_hidden_states = True)
                    prediction_value, confidence_based_prediction = torch.max(outputs.logits, dim=2)
            except:
                break

            i+=1

    reconstruction_err = np.mean(reconstruction_error, axis=0)
    run["recons/average_mse_additive_masking_reconstruction"].log(list(reconstruction_err))
    classification_err = 1 - np.mean(classification_acc, axis=0)
    run["recons/average_classification_error_additive_masking"].log(list(classification_err))
    
    average_errors = []
    for perc in mask_percentages:
        reconstruction_errors = []
        correct_random_pred = 0
        tot_sample = 0
        for x in sample_list:
            print('random approach - event:', x)
            q = torch.from_numpy(quantizes[x]).to(device)
            index = torch.from_numpy(indices[x]).to(device)
            q = torch.reshape(q, (1, q.size(dim=0), q.size(dim=1)))
            
            with torch.no_grad():
                q_masked, index_masked, mask_pattern = random_mask(q, index , n_sample, n_token, perc)                                
                q_masked = q_masked.to(device)
                index_masked = index_masked.to(device)
                print('mask percentage: ', perc)
                print('quantize shape:', q_masked.shape)
                print('index shape:', index_masked.shape)
                print('mask shape:', mask_pattern.shape)
                np.save(f'/home/abghamtm/work/masking_comparison/masking-reconstruction_pattern/random_mask_pattern_{x}_MaskPercentage={perc}.npy', mask_pattern)

            #Fill in predicted tokens
            with torch.no_grad():
                outputs = model_distil(inputs_embeds = q_masked, output_hidden_states = True)
                confidence_based_prediction = torch.argmax(outputs.logits, dim=2)
                confidence_based_recons_index = index
                for p in range(0,n_token):
                    if(mask_pattern[p]):
                        #confidence_based_recons_index[p] = confidence_based_prediction.detach().cpu().numpy()[0][p] 
                        confidence_based_recons_index[p] = confidence_based_prediction[0][p] 
                
                #Reconstruct with distil predictions
                confidence_based_recons_index = confidence_based_recons_index.to(device)
                distil_out = model_vqvae.decode_code(torch.reshape(confidence_based_recons_index, (1,length,length)).to(device))

                #Reconstruct Original
                vqvae_out = model_vqvae.decode(torch.from_numpy(quant_b[x]).to(device)) #torch.reshape(torch.from_numpy(indices[x]), (1,length,length)).to(device)
                index_masked_forvis = index.clone()
                index_masked_forvis[mask_pattern]=0
                vqvae_masked_out = model_vqvae.decode_code(torch.reshape(index_masked_forvis, (1,length,length)).to(device))
                print('original image shape after decoding:', vqvae_out.shape)
                print('Masked image w/o transformer shape after decoding:', vqvae_masked_out.shape)
                print('Masked image w transformer shape after decoding', distil_out.shape)

                # Label outputs
                vqvae_out = vqvae_out.unsqueeze(0)
                vqvae_img = preprocess(vqvae_out)
                vqvae_img = vqvae_img.to(device)
                vqvae_img_prob = classifier(vqvae_img)
                _, vqvae_img_label = torch.max(vqvae_img_prob, 1)

                # vqvae_masked_out = vqvae_masked_out.unsqueeze(0)
                vqvae_masked_img = preprocess(vqvae_masked_out)
                vqvae_masked_img = vqvae_masked_img.to(device)
                vqvae_masked_img_prob = classifier(vqvae_masked_img)
                _, vqvae_masked_img_label = torch.max(vqvae_masked_img_prob, 1)
                
                rand_mask_img = preprocess(distil_out)
                rand_mask_img = rand_mask_img.to(device)
                rand_mask_img_prob = classifier(rand_mask_img)
                _, rand_mask_img_label = torch.max(rand_mask_img_prob, 1)
                correct_random_pred += (rand_mask_img_label == vqvae_img_label).sum().item()
                tot_sample += 1

                percentage = int(perc*100)
                img_list = [vqvae_masked_out, distil_out]
                label_list = [vqvae_masked_img_label.item(), rand_mask_img_label.item()]
                for ii, (img, label) in enumerate(zip(img_list, label_list)):
                    # Move tensor to CPU and convert to NumPy array
                    img_np = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
                    
                    plt.figure()
                    plt.imshow(img_np, cmap='viridis')
                    # plt.xlabel(label)
                    # plt.colorbar()
                    if ii==0:
                        plt.savefig(f'image/recons/comparison/{str(x).zfill(5)}_MaskPercentage={percentage}_RandomMasking_label={label}.png')
                    else:
                        plt.savefig(f'image/recons/comparison/{str(x).zfill(5)}_MaskPercentage={percentage}_RandomMaskingWithTransformer_label={label}.png')
                    plt.close()
            
            recon_loss = criterion(distil_out, vqvae_out)
            reconstruction_errors.append(recon_loss.item())
        run["recons/average_mse_random_masking_reconstruction"].log(np.mean(reconstruction_errors))
        average_errors.append(np.mean(reconstruction_errors))
        pred_acc_random_mask = correct_random_pred/tot_sample
        pred_err_random_mask = 1-pred_acc_random_mask
        run["recons/average_classification_accuracy_random"].log(pred_err_random_mask)

    selective_recons_avg_err = []
    for perc in mask_percentages:
        reconstruction_errors = []
        correct_grad_pred = 0
        tot_sample = 0
        for x in sample_list:
            print('selective approach - event:', x)
            q = torch.from_numpy(quantizes[x]).to(device)
            index = torch.from_numpy(indices[x]).to(device)
            q = torch.reshape(q, (1, q.size(dim=0), q.size(dim=1)))
            
            with torch.no_grad():
                q_masked, index_masked, mask = selective_masking(model_distil, q, index ,perc)                              
                q_masked = q_masked.to(device)
                index_masked = index_masked.to(device)
                print('mask percentage: ', perc)
                print('quantize shape:', q_masked.shape)
                print('index shape:', index_masked.shape)
                print('mask shape:', mask_pattern.shape)
                np.save(f'/home/abghamtm/work/masking_comparison/masking-reconstruction_pattern/selective_mask_pattern_{x}_RandomMasking={perc}.npy', mask_pattern)

            with torch.no_grad():
                outputs = model_distil(inputs_embeds = q_masked, output_hidden_states = True)
                confidence_based_prediction = torch.argmax(outputs.logits, dim=2)
                confidence_based_recons_index = index
                for p in range(0,n_token):
                    if(mask[p]):
                        confidence_based_recons_index[p] = confidence_based_prediction[0][p] 
                
                #Reconstruct with distil predictions
                confidence_based_recons_index = confidence_based_recons_index.to(device)
                distil_out = model_vqvae.decode_code(torch.reshape(confidence_based_recons_index, (1,length,length)).to(device))

                #Reconstruct Original
                vqvae_out = model_vqvae.decode(torch.from_numpy(quant_b[x]).to(device))
                index_masked_forvis = index.clone()
                index_masked_forvis[mask]=0
                vqvae_masked_out = model_vqvae.decode_code(torch.reshape(index_masked_forvis, (1,length,length)).to(device))
                print('original image shape after decoding:', vqvae_out.shape)
                print('Masked image w/o transformer shape after decoding:', vqvae_masked_out.shape)
                print('Masked image w transformer shape after decoding', distil_out.shape)
                
                # Label outputs
                vqvae_out = vqvae_out.unsqueeze(0)
                vqvae_img = preprocess(vqvae_out)
                vqvae_img = vqvae_img.to(device)
                vqvae_img_prob = classifier(vqvae_img)
                _, vqvae_img_label = torch.max(vqvae_img_prob, 1)

                # vqvae_masked_out = vqvae_masked_out.unsqueeze(0)
                vqvae_masked_img = preprocess(vqvae_masked_out)
                vqvae_masked_img = vqvae_masked_img.to(device)
                vqvae_masked_img_prob = classifier(vqvae_masked_img)
                _, vqvae_masked_img_label = torch.max(vqvae_masked_img_prob, 1)
                
                grad_mask_img = preprocess(distil_out)
                grad_mask_img = grad_mask_img.to(device)
                grad_mask_img_prob = classifier(grad_mask_img)
                _, grad_mask_img_label = torch.max(grad_mask_img_prob, 1)
                correct_grad_pred += (grad_mask_img_label == vqvae_img_label).sum().item()
                tot_sample += 1

                percentage = int(perc*100)
                img_list = [vqvae_masked_out, distil_out]
                label_list = [vqvae_masked_img_label.item(), grad_mask_img_label.item()]
                for ii, (img, label) in enumerate(zip(img_list, label_list)):
                    # Move tensor to CPU and convert to NumPy array
                    img_np = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
                    
                    plt.figure()
                    plt.imshow(img_np, cmap='viridis')
                    # plt.xlabel(label)
                    # plt.colorbar()
                    if ii==0:
                        plt.savefig(f'image/recons/comparison/{str(x).zfill(5)}_MaskPercentage={percentage}_SelectiveMasking_label={label}.png')
                    else:
                        plt.savefig(f'image/recons/comparison/{str(x).zfill(5)}_MaskPercentage={percentage}_SelectiveMaskingWithTransformer_label={label}.png')
                    plt.close()                

            recon_loss = criterion(distil_out, vqvae_out)
            reconstruction_errors.append(recon_loss.item())
        run["recons/average_mse_selective_masking_reconstruction"].log(np.mean(reconstruction_errors))
        selective_recons_avg_err.append(np.mean(reconstruction_errors))
        pred_acc_grad_mask = correct_grad_pred/tot_sample
        pred_err_grad_mask = 1-pred_acc_grad_mask
        run["recons/average_classification_accuracy_selective"].log(pred_err_grad_mask)

    # print(sample_list)

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
