import os, yaml, argparse, json, torch, mlflow
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from vqvae import FlatVQVAE
from data_utils import CustomImageNetDataV2, CustomLoader
from gpu_utils import select_gpus
import torch.distributed as dist
import numpy as np
from PIL import Image
from tqdm import tqdm

# ---------- Prioritize Task -----------
os.nice(19)

# ---------- Config & CLI ----------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    return parser.parse_args()

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# ---------- Distributed Setup ----------
def setup_distributed():
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    return rank, world_size, local_rank

# ---------- Loader ----------
def get_loader(dataset, batch_size, shuffle, distributed, world_size=None, rank=None):
    return CustomLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        distributed=distributed, world_size=world_size, rank=rank).data_loader

# ---------- Load Best Params ----------
def load_best_params(model_path):
    for fname in os.listdir(model_path):
        if fname.startswith("best_vqvae_params_trial_") and fname.endswith(".json"):
            with open(os.path.join(model_path, fname), 'r') as f:
                return json.load(f)
    return None

# ---------- Load Best Model ----------
def load_best_model_path(model_path, epoch):
    return os.path.join(model_path, f"model_epoch_{epoch}_vqvae_80x80_codebook_144x456.pth")

# ---------- Main ----------
def main():
    args = parse_args()
    config = load_config(args.config)
    selected_gpu_ids, world_size = select_gpus(config['multiprocessing']['gpu'])
    rank, _, local_rank = setup_distributed()
    device = torch.device(f"cuda:{selected_gpu_ids[local_rank]}")

    # Load datasets
    path_cfg = config['path']
    test_set = CustomImageNetDataV2(image_dir=path_cfg['image_net_test'], image_type='original', folder_label='int_id')

    model_path = path_cfg['vqvae_model']
    os.makedirs(model_path, exist_ok=True)

    # Load best parameters
    best_params = load_best_params(model_path)
    if best_params is None:
        print("⚠️ No best params found. Using defaults from config.")
        best_params = config['params']['vqvae']

    # Extract hyperparameters
    batch_size = best_params['batch_size']
    latent_loss_weight = best_params['latent_loss_weight']
    diversity_loss_weight = best_params['diversity_loss_weight']
    num_epochs = best_params['num_epochs']
    lr = best_params['lr']
    weight_decay = best_params['weight_decay']

     # Load best model checkpoint
    model_ckpt = load_best_model_path(model_path, epoch = num_epochs)
    model = FlatVQVAE().to(device)
    if dist.is_initialized():
        model = DDP(model, device_ids=[device.index])
    model.load_state_dict(torch.load(model_ckpt))
    recon_criterion = nn.MSELoss()

    # ---------- MLflow Logging ----------
    if rank == 0:
        mlflow.start_run(run_name="vqvae_test_and_reconstructing")
        mlflow.log_params({
            "batch_size": batch_size,
            "latent_loss_weight": latent_loss_weight,
            "diversity_loss_weight": diversity_loss_weight,
            "num_epochs": num_epochs,
            "lr": lr,
            "weight_decay": weight_decay
        })

    # ---------- Test Evaluation ----------
    test_loader = get_loader(test_set, batch_size=batch_size, shuffle=False, distributed=False)
    model.eval()
    indices = [] # indices of all images - latent space
    quantizes = [] # codebooks of all images
    labels = [] # labels of all images
    test_loss = 0.0
    with torch.no_grad():
        for j, (inputs, labels) in tqdm(enumerate(test_loader)):
            inputs = inputs.to(device)
            recon, latent_loss, diversity_loss, _ = model(inputs)
            recon_loss = recon_criterion(recon, inputs)
            loss = recon_loss + latent_loss_weight * latent_loss + diversity_loss_weight * diversity_loss
            test_loss += loss.item() * inputs.size(0)

            ## ----------- store codebook, latent space, and corresponding labels -------------
            quant_b, _, id_b, _, _ = model.encode(inputs)
            outputs = model.decode(quant_b)
            indices.append(id_b.cpu())
            quantizes.append(quant_b.cpu())
            labels.extend(labels.cpu().numpy())
            ### ---------- save reconstructed images ----------------
            for idx, (label, out) in enumerate(zip(labels, outputs)):
                print(label)
                class_folder = os.path.join(path_cfg['recnstructed_imge'], str(label.item()))
                os.makedirs(class_folder, exist_ok=True)
                save_file = os.path.join(class_folder, f"{label.item()}_{j * test_loader.batch_size + idx + 1:05d}.png")
                Image.fromarray(out).save(save_file) 

    # Concatenate all indices into a single tensor and save it
    indices_tensor = torch.cat(indices, dim=0)
    quantizes_tensor = torch.cat(quantizes, dim=0)
    labels = np.array(labels)

    indices_path = os.path.join(args.save_path_models, f"latent_space_vqvae_80x80_codebook_144x456.npy")
    quantized_path = os.path.join(args.save_path_models, f"codebook_vqvae_80x80_codebook_144x456.npy")
    np.save(os.path.join(args.save_path_models, f'labels.npy'), labels)

    np.save(indices_path, indices_tensor.numpy())
    np.save(quantized_path, quantizes_tensor.numpy())

    test_loss /= len(test_loader.dataset)
    print(f"✅ Final Test Loss: {test_loss:.4f}")
    if rank == 0:
        mlflow.log_metric("test_loss", test_loss)
        mlflow.end_run()


