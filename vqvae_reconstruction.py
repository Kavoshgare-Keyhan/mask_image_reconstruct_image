import os, yaml, argparse, json, torch, mlflow, subprocess
from mlflow.tracking import MlflowClient
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

# ---------- Load Params ----------
def load_params(model_path):
    '''
    Load initial parameters from the config file used during training if there is no retrieved best parameters from MLflow.
    '''
    for fname in os.listdir(model_path):
        if fname.startswith("best_vqvae_params_trial_") and fname.endswith(".json"):
            with open(os.path.join(model_path, fname), 'r') as f:
                return json.load(f)
    return None

# ---------- MLflow Retrieve Best Epoch ----------
def get_params_from_mlflow():
    '''
    Retrieve the parameters either best or preset by user along with best epoch number from MLflow logged parameters.
    '''
    client = MlflowClient()
    runs = client.search_runs(experiment_ids=["0"])
    for run in runs:
    #     if run.data.tags.get("model_type") == "vqvae_training":
    #         return run.data.params, int(run.data.params['best_epoch'])
    # raise ValueError("No suitable MLflow run found for VQ-VAE training.")
        params = run.data.params
        val_loss = run.data.metrics.get("val_loss", float("inf"))
    return params, val_loss

# ---------- Load Best Model Path----------
def load_best_model_path(model_path, epoch):
    return os.path.join(model_path, f"model_epoch_{epoch}_vqvae_80x80_codebook_144x456.pth")

# Robust checkpoint loading (handles 'module.' prefix differences)
def load_checkpoint_into(model, ckpt_path, map_location=None):
    state = torch.load(ckpt_path, map_location=map_location)
    # If a full checkpoint dict (with 'state_dict'), prioritize that:
    if isinstance(state, dict) and 'state_dict' in state:
        state_dict = state['state_dict']
    else:
        state_dict = state

    # If model is DDP, model.module is actual module; get its state_dict keys to decide
    target_module = model.module if hasattr(model, 'module') else model

    # If keys in checkpoint all start with "module." and target keys don't, strip prefixes
    ck_keys = list(state_dict.keys())
    if len(ck_keys) > 0 and ck_keys[0].startswith('module.') and not any(k.startswith('module.') for k in target_module.state_dict().keys()):
        new_state = {}
        for k, v in state_dict.items():
            new_state[k.replace('module.', '', 1)] = v
        state_dict = new_state

    # Vice versa: if checkpoint keys don't have 'module.' but target expects them, add prefix
    target_keys = list(target_module.state_dict().keys())
    if len(target_keys) > 0 and target_keys[0].startswith('module.') and not ck_keys[0].startswith('module.'):
        new_state = {}
        for k, v in state_dict.items():
            new_state['module.' + k] = v
        state_dict = new_state

    # Finally load (use strict=False to be tolerant)
    target_module.load_state_dict(state_dict, strict=False)

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

    # Load parameters
    try:
        params_mlflow, val_loss = get_params_from_mlflow()
        epoch_least_val_loss = int(params_mlflow['best_epoch'])
        batch_size = int(params_mlflow['batch_size'])
        latent_loss_weight = float(params_mlflow['latent_loss_weight'])
        diversity_loss_weight = float(params_mlflow['diversity_loss_weight'])
    except:
        trained_params = config['params']['vqvae']
        # Extract hyperparameters
        batch_size = trained_params['batch_size']
        latent_loss_weight = trained_params['latent_loss_weight']
        diversity_loss_weight = trained_params['diversity_loss_weight']
        epoch_least_val_loss = trained_params['num_epochs'] - 1
        _, val_loss = get_params_from_mlflow()
        # lr = float(trained_params['lr'])
        # weight_decay = float(trained_params['weight_decay'])

    # Load best model checkpoint
    model_ckpt = load_best_model_path(model_path, epoch = epoch_least_val_loss)
    model = FlatVQVAE().to(device)
    load_checkpoint_into(model, model_ckpt, map_location=device)

    if dist.is_initialized():
        model = DDP(model, device_ids=[device.index])

    recon_criterion = nn.MSELoss()

    # ---------- MLflow Logging ----------
    if rank == 0:
        # Create or set experiment (vqvae_experiment will get its own folder under mlruns)
        mlflow.set_experiment("reconstruction_experiment")

        # Get current git commit hash for reproducibility
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()

        # Start run with descriptive name and tags
        run_context = mlflow.start_run(
            run_name="vqvae_reconstruction_run",
            tags={"model_type": "vqvae_reconstruction", "git_commit": commit}
        )
        mlflow.log_params({
            "batch_size": batch_size,
            "val_loss": val_loss,
            "chosen_model": f'model_epoch_{epoch_least_val_loss}_vqvae_80x80_codebook_144x456.pth'
        })

    # ---------- Test Evaluation ----------
    train_loader = get_loader(train_set, batch_size=batch_size, shuffle=True, distributed=dist.is_initialized(), world_size=world_size, rank=rank)
    # test_loader = get_loader(test_set, batch_size=batch_size, shuffle=False, distributed=False)
    model.eval()
    tr_indices = [] # indices of all images - latent space
    tr_quantizes = [] # codebooks of all images
    tr_labels = [] # labels of all images

    # create base_model reference to underlying module when needed
    base_model = model.module if hasattr(model, 'module') else model

    with torch.no_grad():
        for j, (inputs, labels) in tqdm(enumerate(train_loader)):
            inputs = inputs.to(device)
            recon, latent_loss, diversity_loss, _ = model(inputs)
            recon_loss = recon_criterion(recon, inputs)
            loss = recon_loss + latent_loss_weight * latent_loss + diversity_loss_weight * diversity_loss
            # test_loss += loss.item() * inputs.size(0)

            ## ----------- store codebook, latent space, and corresponding labels -------------
            quant_b, _, id_b, _, _ = base_model.encode(inputs)
            outputs = base_model.decode(quant_b)
            tr_indices.append(id_b.cpu())
            tr_quantizes.append(quant_b.cpu())
            tr_labels.extend(labels.cpu().numpy())
            ### ---------- save reconstructed images ----------------
            for idx, (label, out) in enumerate(zip(labels, outputs)):
                print(label)
                tr_class_folder = os.path.join(path_cfg['recnstructed_imge']['train'], str(label.item()))
                os.makedirs(tr_class_folder, exist_ok=True)
                save_tr_file = os.path.join(tr_class_folder, f"{label.item()}_{j * train_loader.batch_size + idx + 1:05d}.npy")
                np.save(save_tr_file, out.cpu().numpy())


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
            quant_b, _, id_b, _, _ = base_model.encode(inputs)
            outputs = base_model.decode(quant_b)
            val_indices.append(id_b.cpu())
            val_quantizes.append(quant_b.cpu())
            val_labels.extend(labels.cpu().numpy())
            ### ---------- save reconstructed images ----------------
            for idx, (label, out) in enumerate(zip(labels, outputs)):
                print(label)
                val_class_folder = os.path.join(path_cfg['recnstructed_imge']['val'], str(label.item()))
                os.makedirs(val_class_folder, exist_ok=True)
                save_val_file = os.path.join(val_class_folder, f"{label.item()}_{j * val_loader.batch_size + idx + 1:05d}.npy")
                np.save(save_val_file, out.cpu().numpy())

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

if __name__ == "__main__":
    main()    


