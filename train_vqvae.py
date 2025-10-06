import os, yaml, argparse, json, torch, mlflow
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from vqvae import FlatVQVAE
from data_utils import CustomImageNetDataV2, CustomLoader
from gpu_utils import select_gpus
import torch.distributed as dist

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
def load_best_model_path(model_path):
    for fname in os.listdir(model_path):
        if fname.startswith("best_vqvae_model_trial_") and fname.endswith(".pth"):
            return os.path.join(model_path, fname)
    return None

# ---------- Main ----------
def main():
    args = parse_args()
    config = load_config(args.config)
    selected_gpu_ids, world_size = select_gpus(config['multiprocessing']['gpu'])
    rank, _, local_rank = setup_distributed()
    device = torch.device(f"cuda:{selected_gpu_ids[local_rank]}")

    # Load datasets
    path_cfg = config['path']
    train_set = CustomImageNetDataV2(image_dir=path_cfg['image_net_train'], image_type='original', folder_label='int_id')
    val_set = CustomImageNetDataV2(image_dir=path_cfg['image_net_val'], image_type='original', folder_label='int_id')
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

    # Train on train+val
    full_set = torch.utils.data.ConcatDataset([train_set, val_set])
    full_loader = get_loader(full_set, batch_size=batch_size, shuffle=True, distributed=dist.is_initialized(), world_size=world_size, rank=rank)
    train_loader = get_loader(train_set, batch_size=batch_size, shuffle=True, distributed=dist.is_initialized(), world_size=world_size, rank=rank)
    val_loader = get_loader(val_set, batch_size=batch_size, shuffle=True, distributed=dist.is_initialized(), world_size=world_size, rank=rank)

    # Load best model checkpoint
    model_ckpt = load_best_model_path(model_path)
    model = FlatVQVAE().to(device)
    if dist.is_initialized():
        model = DDP(model, device_ids=[device.index])
    if model_ckpt:
        model.load_state_dict(torch.load(model_ckpt))

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    recon_criterion = nn.MSELoss()

    # ---------- MLflow Logging ----------
    if rank == 0:
        mlflow.start_run(run_name="vqvae_final_training")
        mlflow.log_params({
            "batch_size": batch_size,
            "latent_loss_weight": latent_loss_weight,
            "diversity_loss_weight": diversity_loss_weight,
            "num_epochs": num_epochs,
            "lr": lr,
            "weight_decay": weight_decay
        })

    model.train()
    if model_ckpt:
        for epoch in range(num_epochs):
            train_loss = 0.0
            for inputs, _ in full_loader:
                inputs = inputs.to(device)
                optimizer.zero_grad()
                recon, latent_loss, diversity_loss, _ = model(inputs)
                recon_loss = recon_criterion(recon, inputs)
                loss = recon_loss + latent_loss_weight * latent_loss + diversity_loss_weight * diversity_loss
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)

            train_loss /= len(full_loader.dataset)
            if rank == 0:
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}")
            save_path_model = os.path.join(model_path, f"model_epoch_{epoch}_vqvae_80x80_codebook_144x456.pth")
            torch.save(model.state_dict(), save_path_model)
    else:
        least_val_loss = float('inf')
        for epoch in range(num_epochs):
            train_loss = 0.0
            for inputs, _ in train_loader:
                inputs = inputs.to(device)
                optimizer.zero_grad()
                recon, latent_loss, diversity_loss, _ = model(inputs)
                recon_loss = recon_criterion(recon, inputs)
                loss = recon_loss + latent_loss_weight * latent_loss + diversity_loss_weight * diversity_loss
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)

            train_loss /= len(train_loader.dataset)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, _ in val_loader:
                    inputs = inputs.to(device)
                    recon, latent_loss, diversity_loss, _ = model(inputs)
                    recon_loss = recon_criterion(recon, inputs)
                    loss = recon_loss + latent_loss_weight * latent_loss + diversity_loss_weight * diversity_loss
                    val_loss += loss.item() * inputs.size(0)

            val_loss /= len(val_loader.dataset)
            if rank == 0:
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            if val_loss < least_val_loss:
                least_val_loss = val_loss
            else:
                print(f'Least Validation Loss is {least_val_loss: .4f} and belongs to epoch {epoch}')
                save_path_model = os.path.join(model_path, f"model_epoch_{epoch}_vqvae_80x80_codebook_144x456.pth")
                torch.save(model.state_dict(), save_path_model)
                if rank == 0:
                    mlflow.log_artifact(save_path_model)

    if rank == 0:
        mlflow.end_run()

if __name__ == "__main__":
    main()
