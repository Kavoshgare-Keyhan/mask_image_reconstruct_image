#!/usr/bin/env python3
"""
Memory-safe VQ-VAE reconstruction script.
- Appends indices & quantized vectors + labels to per-rank HDF5 files.
- Restores MLflow logging (experiment/run/params).
- DDP-compatible. Does not accumulate tensors in Python lists.
- Does NOT save reconstructed images (user choice).
"""

import os, yaml, argparse, torch, subprocess, h5py, mlflow
from mlflow.tracking import MlflowClient
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from vqvae import FlatVQVAE
# use the uploaded/modified data_utils (path: /mnt/data/data_utils.py)
from data_utils import CustomImageNetDataV2, CustomLoader
from gpu_utils import select_gpus
import torch.distributed as dist
from tqdm import tqdm

# ---------- priority niceness ----------
try:
    os.nice(19)
except Exception:
    pass

# ---------- CLI & config ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='config.yaml')
    return p.parse_args()

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# ---------- distributed setup ----------
def setup_distributed(selected_gpu_ids):
    # tries to respect torchrun env vars; otherwise fall back to single-process
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', len(selected_gpu_ids)))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    if 'RANK' not in os.environ:
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['LOCAL_RANK'] = str(local_rank)

    torch.cuda.set_device(selected_gpu_ids[local_rank])
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    return rank, world_size, local_rank

# ---------- checkpoint loader ----------
def load_checkpoint_into(model, ckpt_path, map_location=None):
    state = torch.load(ckpt_path, map_location=map_location)
    if isinstance(state, dict) and 'state_dict' in state:
        state_dict = state['state_dict']
    else:
        state_dict = state

    target_module = model.module if hasattr(model, 'module') else model

    ck_keys = list(state_dict.keys())
    # handle module. prefixes
    if len(ck_keys) > 0 and ck_keys[0].startswith('module.') and not any(k.startswith('module.') for k in target_module.state_dict().keys()):
        new_state = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
        state_dict = new_state

    target_keys = list(target_module.state_dict().keys())
    if len(target_keys) > 0 and target_keys[0].startswith('module.') and not ck_keys[0].startswith('module.'):
        new_state = {'module.' + k: v for k, v in state_dict.items()}
        state_dict = new_state

    target_module.load_state_dict(state_dict, strict=False)

# ---------- HDF5 append helper ----------
class H5Appender:
    def __init__(self, path):
        self.path = path
        ddir = os.path.dirname(path)
        if ddir:
            os.makedirs(ddir, exist_ok=True)
        # open in append mode
        self.f = h5py.File(path, 'a')

    def append(self, name, arr):
        # arr: numpy array with shape (N, ...)
        if name not in self.f:
            maxshape = (None,) + arr.shape[1:]
            # create dataset with chunking for better append performance
            self.f.create_dataset(name, data=arr, maxshape=maxshape, chunks=True)
        else:
            ds = self.f[name]
            old = ds.shape[0]
            ds.resize((old + arr.shape[0],) + ds.shape[1:])
            ds[old:old + arr.shape[0]] = arr

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass

# ---------- MLflow helpers ----------
def start_mlflow_if_rank0(rank, config, params_log):
    """
    Start MLflow run and log params if rank==0.
    Returns run context (None if not rank0).
    """
    if rank != 0:
        return None

    mlflow.set_experiment(config.get('mlflow_experiment', 'reconstruction_experiment'))
    commit = None
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()
    except Exception:
        pass

    run = mlflow.start_run(run_name="vqvae_reconstruction_run", tags={"model_type": "vqvae_reconstruction", "git_commit": commit})
    # log any available params
    try:
        mlflow.log_params(params_log)
    except Exception as e:
        print("⚠️ mlflow.log_params failed:", e)
    return run

def end_mlflow_if_rank0(rank, run):
    if rank != 0 or run is None:
        return
    try:
        mlflow.end_run()
    except Exception:
        pass

# ---------- main ----------
def main():
    args = parse_args()
    cfg = load_config(args.config)

    selected_gpu_ids = select_gpus(cfg['multiprocessing']['gpu'])
    rank, world_size, local_rank = setup_distributed(selected_gpu_ids)
    device = torch.device(f"cuda:{selected_gpu_ids[local_rank]}")

    # path config
    path_cfg = cfg['path']
    model_path = path_cfg['model']['vqvae']
    os.makedirs(model_path, exist_ok=True)

    # prepare datasets
    try:
        train_set = CustomImageNetDataV2(image_dir=path_cfg['image_net_train'], image_type='original', folder_label='word_net_id')
        val_set = CustomImageNetDataV2(image_dir=path_cfg['image_net_val'], image_type='original', folder_label='word_net_id')
    except:
        train_set = CustomImageNetDataV2(image_dir=path_cfg['image_net_train_local'], image_type='original', folder_label='word_net_id')
        val_set = CustomImageNetDataV2(image_dir=path_cfg['image_net_val_local'], image_type='original', folder_label='word_net_id')

    # get params: try mlflow fallback to config
    batch_size = cfg['params']['vqvae'].get('batch_size', 32)
    best_epoch = cfg['params']['vqvae'].get('num_epochs', 1) - 1

    # If mlflow has run with saved params, try to use it (non-fatal)
    try:
        client = MlflowClient()
        runs = client.search_runs(experiment_ids=["0"]) or []
        if runs:
            params_mlflow = runs[0].data.params
            batch_size = int(params_mlflow.get('batch_size', batch_size))
            best_epoch = int(params_mlflow.get('best_epoch', best_epoch))
    except Exception:
        pass

    # load model checkpoint
    model_ckpt = os.path.join(model_path, f"model_epoch_{best_epoch}_vqvae_80x80_codebook_144x456.pth")
    model = FlatVQVAE().to(device)
    load_checkpoint_into(model, model_ckpt, map_location=device)

    if dist.is_initialized():
        model = DDP(model, device_ids=[device.index])

    base_model = model.module if hasattr(model, 'module') else model

    # restore mlflow run and log params (rank 0)
    params_to_log = {
        "batch_size": batch_size,
        "best_epoch": best_epoch,
        "vqvae_ckpt": os.path.basename(model_ckpt)
    }
    mlflow_run = start_mlflow_if_rank0(rank, cfg, params_to_log)

    # dataloader conservative settings (configurable)
    recon_cfg = cfg.get('reconstruction', {})
    num_workers = recon_cfg.get('num_workers', 2)
    pin_memory = recon_cfg.get('pin_memory', False)

    # use our CustomLoader (which accepts threads and pin_memory)
    train_loader = CustomLoader(train_set, batch_size=batch_size, threads=num_workers,
                                shuffle=False, distributed=dist.is_initialized(),
                                world_size=world_size, rank=rank, pin_memory=pin_memory).data_loader

    val_loader = CustomLoader(val_set, batch_size=batch_size, threads=num_workers,
                              shuffle=False, distributed=dist.is_initialized(),
                              world_size=world_size, rank=rank, pin_memory=pin_memory).data_loader

    recon_criterion = nn.MSELoss()

    # output HDF5 per-rank
    out_dir = path_cfg.get('reconstructed_storage', os.path.join(model_path, 'reconstruction_outputs'))
    os.makedirs(out_dir, exist_ok=True)
    train_h5 = H5Appender(os.path.join(out_dir, f"train_rank{rank}.h5"))
    val_h5 = H5Appender(os.path.join(out_dir, f"val_rank{rank}.h5"))

    # ---- process train set ----
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"[rank {rank}] train")):
            inputs = inputs.to(device, non_blocking=pin_memory)
            # forward + encode
            # some VQ-VAE may return recon first; the exact call matches your vqvae.encode
            recon, latent_loss, diversity_loss, _ = model(inputs)
            quant_b, _, id_b, _, _ = base_model.encode(inputs)

            # move minimal required arrays to CPU & numpy immediately
            id_np = id_b.cpu().numpy()
            quant_np = quant_b.cpu().numpy()
            labels_np = labels.numpy()

            # append to HDF5 (per-rank)
            train_h5.append('indices', id_np)
            train_h5.append('quantized', quant_np)
            train_h5.append('labels', labels_np.reshape((-1,)))

            # cleanup to free memory immediately
            del id_b, quant_b, id_np, quant_np, labels_np, recon, latent_loss, diversity_loss
            torch.cuda.empty_cache()

    # ---- process val set ----
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(val_loader, desc=f"[rank {rank}] val")):
            inputs = inputs.to(device, non_blocking=pin_memory)
            quant_b, _, id_b, _, _ = base_model.encode(inputs)

            id_np = id_b.cpu().numpy()
            quant_np = quant_b.cpu().numpy()
            labels_np = labels.numpy()

            val_h5.append('indices', id_np)
            val_h5.append('quantized', quant_np)
            val_h5.append('labels', labels_np.reshape((-1,)))

            del id_b, quant_b, id_np, quant_np, labels_np
            torch.cuda.empty_cache()

    # close files
    train_h5.close()
    val_h5.close()

    # end mlflow
    end_mlflow_if_rank0(rank, mlflow_run)

if __name__ == "__main__":
    main()
