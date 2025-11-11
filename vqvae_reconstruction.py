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
def setup_distributed(selected_gpu_ids):
    # Try to get values from torchrun or environment
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', len(selected_gpu_ids)))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    # Fallback for pure python run
    if 'RANK' not in os.environ:
        print("⚠️ Environment variables for distributed training not found. Assuming single-process fallback.")
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['LOCAL_RANK'] = str(local_rank)

    torch.cuda.set_device(selected_gpu_ids[local_rank])
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
    selected_gpu_ids = select_gpus(config['multiprocessing']['gpu'])
    rank, world_size, local_rank = setup_distributed(selected_gpu_ids)
    device = torch.device(f"cuda:{selected_gpu_ids[local_rank]}")

    # Load datasets
    path_cfg = config['path']
    train_set = CustomImageNetDataV2(image_dir=path_cfg['image_net_train'], image_type='original', folder_label='word_net_id')
    val_set = CustomImageNetDataV2(image_dir=path_cfg['image_net_val'], image_type='original', folder_label='word_net_id')
    # test_set = CustomImageNetDataV2(image_dir=path_cfg['image_net_test'], image_type='original', folder_label='word_net_id')
    

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
    num_epochs = best_params['num_epochs'] - 1
    lr = float(best_params['lr'])
    weight_decay = float(best_params['weight_decay'])

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
    train_loader = get_loader(train_set, batch_size=batch_size, shuffle=True, distributed=dist.is_initialized(), world_size=world_size, rank=rank)
    # test_loader = get_loader(test_set, batch_size=batch_size, shuffle=False, distributed=False)
    model.eval()
    tr_indices = [] # indices of all images - latent space
    tr_quantizes = [] # codebooks of all images
    tr_labels = [] # labels of all images
    # test_loss = 0.0
    with torch.no_grad():
        for j, (inputs, labels) in tqdm(enumerate(train_loader)):
            inputs = inputs.to(device)
            recon, latent_loss, diversity_loss, _ = model(inputs)
            recon_loss = recon_criterion(recon, inputs)
            loss = recon_loss + latent_loss_weight * latent_loss + diversity_loss_weight * diversity_loss
            # test_loss += loss.item() * inputs.size(0)

            ## ----------- store codebook, latent space, and corresponding labels -------------
            quant_b, _, id_b, _, _ = model.encode(inputs)
            outputs = model.decode(quant_b)
            tr_indices.append(id_b.cpu())
            tr_quantizes.append(quant_b.cpu())
            tr_labels.extend(labels.cpu().numpy())
            ### ---------- save reconstructed images ----------------
            for idx, (label, out) in enumerate(zip(labels, outputs)):
                print(label)
                class_folder = os.path.join(path_cfg['recnstructed_imge']['train'], str(label.item()))
                os.makedirs(class_folder, exist_ok=True)
                save_file = os.path.join(class_folder, f"{label.item()}_{j * train_loader.batch_size + idx + 1:05d}.png")
                Image.fromarray(out.cpu().numpy()).save(save_file) 

    # Concatenate all indices into a single tensor and save it
    tr_indices_tensor = torch.cat(tr_indices, dim=0)
    tr_quantizes_tensor = torch.cat(tr_quantizes, dim=0)
    tr_labels = np.array(tr_labels)

    tr_indices_path = os.path.join(model_path, f"training_latent_space_vqvae_80x80_codebook_144x456.npy")
    tr_quantized_path = os.path.join(model_path, f"training_codebook_vqvae_80x80_codebook_144x456.npy")

    np.save(tr_indices_path, tr_indices_tensor.numpy())
    np.save(tr_quantized_path, tr_quantizes_tensor.numpy())
    np.save(os.path.join(model_path, f'tr_labels.npy'), tr_labels)

    # Repeat all thes steps for validation set
    val_loader = get_loader(val_set, batch_size=batch_size, shuffle=True, distributed=dist.is_initialized(), world_size=world_size, rank=rank)
    model.eval()
    val_indices = [] # indices of all images - latent space
    val_quantizes = [] # codebooks of all images
    val_labels = [] # labels of all images
    # test_loss = 0.0
    with torch.no_grad():
        for j, (inputs, labels) in tqdm(enumerate(val_loader)):
            inputs = inputs.to(device)
            recon, latent_loss, diversity_loss, _ = model(inputs)
            recon_loss = recon_criterion(recon, inputs)
            loss = recon_loss + latent_loss_weight * latent_loss + diversity_loss_weight * diversity_loss
            # test_loss += loss.item() * inputs.size(0)

            ## ----------- store codebook, latent space, and corresponding labels -------------
            quant_b, _, id_b, _, _ = model.encode(inputs)
            outputs = model.decode(quant_b)
            val_indices.append(id_b.cpu())
            val_quantizes.append(quant_b.cpu())
            val_labels.extend(labels.cpu().numpy())
            ### ---------- save reconstructed images ----------------
            for idx, (label, out) in enumerate(zip(labels, outputs)):
                print(label)
                class_folder = os.path.join(path_cfg['recnstructed_imge']['val'], str(label.item()))
                os.makedirs(class_folder, exist_ok=True)
                save_file = os.path.join(class_folder, f"{label.item()}_{j * val_loader.batch_size + idx + 1:05d}.png")
                Image.fromarray(out.cpu().numpy()).save(save_file) 

    # Concatenate all indices into a single tensor and save it
    val_indices_tensor = torch.cat(val_indices, dim=0)
    val_quantizes_tensor = torch.cat(val_quantizes, dim=0)
    val_labels = np.array(val_labels)

    val_indices_path = os.path.join(model_path, f"val_latent_space_vqvae_80x80_codebook_144x456.npy")
    val_quantized_path = os.path.join(model_path, f"val_codebook_vqvae_80x80_codebook_144x456.npy")

    np.save(val_indices_path, val_indices_tensor.numpy())
    np.save(val_quantized_path, val_quantizes_tensor.numpy())
    np.save(os.path.join(model_path, f'val_labels.npy'), val_labels)


    # test_loss /= len(test_loader.dataset)
    # print(f"✅ Final Test Loss: {test_loss:.4f}")
    if rank == 0:
        # mlflow.log_metric("test_loss", test_loss)
        mlflow.end_run()


