# Additive Masking
# Replacement of Masking pattern

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse, math, sys, os, random
import torch
from torch import nn
from torchvision import utils 
import distributed as dist
import matplotlib.pyplot as plt
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
def mask(unmasked_quantizes, unmasked_indices, n_token, unmask_pattern=None):
    mask_pattern = np.random.default_rng().choice([True, False], size=(1,1, n_token), p=[1, 0])
    if unmask_pattern: mask_pattern[0,0,unmask_pattern] = False
    masked = unmasked_quantizes.clone() # shallow copy
    masked[mask_pattern] = 0  # Assuming 0 is the mask token
    indices_masked = unmasked_indices.clone()
    indices_masked[~mask_pattern[0]] = -100 # Assuming -100 is the mask label token
   
    return masked, indices_masked, mask_pattern[0][0]

def extract_min_indices(distil_output, n_token):
    min_values, min_row_indices = torch.min(distil_output.logits, dim=1)
    min_values_flat = min_values.flatten()
    min_row_indices_flat = min_row_indices.flatten()
    # column_indices_flat = torch.arange(456).repeat(1, 1).flatten()
    top_min_values, top_indices = torch.topk(min_values_flat, n_token, largest=False)
    top_row_indices = min_row_indices_flat[top_indices]
    return list(top_row_indices.cpu().numpy())

def reconstruct(distil_output, index, n_token, length, mask_pattern, model_vqvae, device):
    confidence_based_prediction = torch.argmax(distil_output.logits, dim=2)
    confidence_based_recons_index = index
    for p in range(0,n_token):
        if(mask_pattern[p]):
            #confidence_based_recons_index[p] = confidence_based_prediction.detach().cpu().numpy()[0][p] 
            confidence_based_recons_index[p] = confidence_based_prediction[0][p] 
    
    #Reconstruct with distil predictions
    confidence_based_recons_index = confidence_based_recons_index.to(device)
    distil_out = model_vqvae.decode_code(torch.reshape(confidence_based_recons_index, (1,length,length)).to(device))
    return distil_out

def main(args):
    torch.cuda.set_device(1)
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
    classifier.load_state_dict(torch.load('/home/abghamtm/work/masking_comparison/checkpoint/classifier/resnet50/weights_epoch30.pth'))
    classifier.to(device)
    classifier.eval()

    mask_percentages = np.arange(0.1, 1.1, 0.1)
    mask_percentages = np.append(mask_percentages,[.85,.95])
    mask_percentages = np.sort(mask_percentages)
    mask_percentages = mask_percentages[::-1] # reverse the sorted array
    mask_perc_map_indices_length = n_token*mask_percentages

    reconstruction_error = np.zeros((quantizes.shape[0],len(mask_percentages)))
    classification_acc = np.zeros((quantizes.shape[0],len(mask_percentages)))
    # cross_entropy_class_err = []


    criterion = nn.MSELoss()
    for x in range(0, quantizes.shape[0]): 
        print(x)
        q = torch.from_numpy(quantizes[x]).to(device)
        index = torch.from_numpy(indices[x]).to(device)
        q = torch.reshape(q, (1, q.size(dim=0), q.size(dim=1)))
        
        i = 0
        full_indices_list = set(range(400))
        unmask_pattern = []
        while i < len(mask_perc_map_indices_length):
            q_masked, index_masked, mask_pattern = mask(q, index , n_token, unmask_pattern)                                
            q_masked = q_masked.to(device)
            index_masked = index_masked.to(device)
            with torch.no_grad():
                outputs = model_distil(inputs_embeds = q_masked, output_hidden_states = True)
                distil_out = reconstruct(outputs, index, n_token, length, mask_pattern, model_vqvae, device)

                vqvae_out = model_vqvae.decode(torch.from_numpy(quant_b[x]).to(device)) #torch.reshape(torch.from_numpy(indices[x]), (1,length,length)).to(device)
                index_masked_forvis = index.clone()
                index_masked_forvis[mask_pattern]=0
                vqvae_masked_out = model_vqvae.decode_code(torch.reshape(index_masked_forvis, (1,length,length)).to(device))

                # Label outputs
                vqvae_out = vqvae_out.unsqueeze(0)
                vqvae_img = preprocess(vqvae_out)
                vqvae_img = vqvae_img.to(device)
                vqvae_img_prob = classifier(vqvae_img)
                
                add_mask_img = preprocess(distil_out)
                add_mask_img = add_mask_img.to(device)
                add_mask_img_prob = classifier(add_mask_img)

            for j in range(int((mask_perc_map_indices_length[i-1]-mask_perc_map_indices_length[i])//5)):
                top5 = []
                top_row_indices = extract_min_indices(distil_output=outputs, n_token=n_token)
                for ind in top_row_indices:
                    if ind not in unmask_pattern and len(top5)<5:
                        top5.append(ind)
                        unmask_pattern.append(ind)
                if len(top5)<5:
                    n = 5 - len(top5)
                    randomly_selected_indices = random.sample(list(full_indices_list), n)
                    top5.extend(randomly_selected_indices)
                    unmask_pattern.extend(randomly_selected_indices)
                
                full_indices_list = full_indices_list - set(unmask_pattern)
                q_masked, index_masked, mask_pattern = mask(q, index , n_token, top5)                                
                q_masked = q_masked.to(device)
                index_masked = index_masked.to(device)
                with torch.no_grad():
                    outputs = model_distil(inputs_embeds = q_masked, output_hidden_states = True)
            
            recons_loss = criterion(distil_out, vqvae_out)
            reconstruction_error[x,i] = recons_loss.item()
            _, vqvae_img_label = torch.max(vqvae_img_prob, 1)
            _, add_mask_img_label = torch.max(add_mask_img_prob, 1)
            classification_acc[x,i] = (add_mask_img_label == vqvae_img_label).sum().item()

            i+=1

    reconstruction_err = np.mean(reconstruction_error, axis=0)
    run["recons/average_mse_additive_masking_reconstruction"].log(list(reconstruction_err))
    classification_err = 1 - np.mean(classification_acc, axis=0)
    run["recons/average_classification_error_additive_masking"].log(list(classification_err))

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


                