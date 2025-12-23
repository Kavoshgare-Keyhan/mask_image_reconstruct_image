# vqvae_reconstruction.py
import os, yaml, argparse, torch, numpy as np, h5py, subprocess, mlflow
from torch.nn.parallel import DistributedDataParallel as DDP
from vqvae import FlatVQVAE
from data_utils import CustomImageNetDataV2, CustomLoader
from gpu_utils import select_gpus
import torch.distributed as dist
from tqdm import tqdm

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='config.yaml')
    p.add_argument('--gpu_ids', type=int, nargs='+', default=None)
    return p.parse_args()

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def setup_distributed(selected_gpu_ids):
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

def resolve_vqvae_ckpt(cfg, model_path):
    # 1) Prefer best model if present
    best_ckpt = os.path.join(model_path, "best_vqvae_model.pth")
    if os.path.exists(best_ckpt):
        return best_ckpt

    # 2) Fallback to explicit best epoch from config (if provided)
    best_epoch = cfg.get('params', {}).get('vqvae', {}).get('best_model_epoch', None)
    if best_epoch is not None:
        ckpt = os.path.join(model_path, f"model_epoch_{best_epoch}_vqvae.pth")
        if os.path.exists(ckpt):
            return ckpt

    # 3) Final fallback: last epoch by num_epochs - 1
    num_epochs = cfg.get('params', {}).get('vqvae', {}).get('num_epochs', None)
    if num_epochs is None:
        raise FileNotFoundError(
            "No checkpoint found and cfg['params']['vqvae']['num_epochs'] is missing."
        )
    last_epoch_ckpt = os.path.join(model_path, f"model_epoch_{num_epochs - 1}_vqvae.pth")
    if os.path.exists(last_epoch_ckpt):
        return last_epoch_ckpt

    # 4) If nothing exists, raise a clear error
    raise FileNotFoundError(
        f"No VQ-VAE checkpoints found under {model_path}. "
        "Checked: best_vqvae_model.pth, explicit best_model_epoch, and last epoch."
    )

class H5Writer:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = h5py.File(path, 'a')
        # Ensure datasets exist lazily (we'll create on first append)
    def append(self, name, arr):
        arr = np.asarray(arr)
        if name not in self.f:
            maxshape = (None,) + arr.shape[1:]
            self.f.create_dataset(name, data=arr, maxshape=maxshape, chunks=True)
        else:
            ds = self.f[name]
            old = ds.shape[0]
            ds.resize((old + arr.shape[0],) + ds.shape[1:])
            ds[old:old + arr.shape[0]] = arr
    def close(self):
        self.f.close()

def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.gpu_ids is not None: cfg['multiprocessing']['gpu']['gpu_ids'] = args.gpu_ids
    selected_gpu_ids = select_gpus(cfg['multiprocessing']['gpu'])
    rank, world_size, local_rank = setup_distributed(selected_gpu_ids)
    device = torch.device(f"cuda:{selected_gpu_ids[local_rank]}")

    path_cfg = cfg['path']
    model_path = path_cfg['model']['vqvae']
    
    store_dir = path_cfg.get('recnstructed_storage', {}).get('latent_space', '/home/abghamtm/work/mask_image_reconstruct_image/reconstruction/latent_space')
    out_dir = os.path.join(store_dir, 'indices_h5')
    os.makedirs(out_dir, exist_ok=True)

    # prepare datasets
    try:
        train_set = CustomImageNetDataV2(image_dir=path_cfg['image_net_train'], image_type='original', folder_label='word_net_id')
        val_set = CustomImageNetDataV2(image_dir=path_cfg['image_net_val'], image_type='original', folder_label='word_net_id')
    except:
        train_set = CustomImageNetDataV2(image_dir=path_cfg['image_net_train_local'], image_type='original', folder_label='word_net_id')
        val_set = CustomImageNetDataV2(image_dir=path_cfg['image_net_val_local'], image_type='original', folder_label='word_net_id')

    # load pretrained model
    # load model with lowest val loss if exists
    ckpt_path = resolve_vqvae_ckpt(cfg, model_path)
    model = FlatVQVAE().to(device)
    load_checkpoint_into(model, ckpt_path, map_location=device)
    if dist.is_initialized():
        model = DDP(model, device_ids=[device.index])
    base_model = model.module if hasattr(model, 'module') else model

    # dataloaders (do not pin memory by default)
    recon_cfg = cfg.get('reconstruction', {})
    num_workers = recon_cfg.get('num_workers', 2)
    pin_memory = recon_cfg.get('pin_memory', False)

    train_loader = CustomLoader(train_set, batch_size=cfg['params']['vqvae'].get('batch_size', 64),
                                threads=num_workers, shuffle=False, distributed=dist.is_initialized(),
                                world_size=world_size, rank=rank, pin_memory=pin_memory).data_loader
    val_loader = CustomLoader(val_set, batch_size=cfg['params']['vqvae'].get('batch_size', 64),
                              threads=num_workers, shuffle=False, distributed=dist.is_initialized(),
                              world_size=world_size, rank=rank, pin_memory=pin_memory).data_loader

    train_h5 = H5Writer(os.path.join(out_dir, f"train_rank{rank}.h5"))
    val_h5 = H5Writer(os.path.join(out_dir, f"val_rank{rank}.h5"))

    try:
        # MLflow start
        if rank == 0:
            mlflow.set_experiment("vqvae_experiment")
            commit = None
            try:
                commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()
            except Exception:
                pass
            run = mlflow.start_run(run_name="reconstructing_latent_space", tags={"git_commit": commit})
            print("Experiment ID:", run.info.experiment_id) 
            print("Run ID:", run.info.run_id)
            mlflow.log_params({
                "batch_size": cfg['params']['vqvae']['batch_size'], "lr": cfg['params']['vqvae']['lr'], "weight_decay": cfg['params']['vqvae']['weight_decay'],
                "num_epochs": cfg['params']['vqvae']['num_epochs'], "latent_loss_weight": cfg['params']['vqvae']['latent_loss_weight'],
                "diversity_loss_weight": cfg['params']['vqvae']['diversity_loss_weight']
            })
        else:
            print(f"rank is {rank}, no mlflow logging")
            run = None
        # ---- reconstruction loop ----
        model.eval()
        with torch.no_grad():
            # train loop: encode and write indices + labels only
            for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"train_rank{rank}")):
                inputs = inputs.to(device)
                _, _, id_b, _ = base_model.encode(inputs)   # id_b is CPU or GPU tensor of ints
                id_np = id_b.cpu().numpy().astype(np.int32)
                labels_np = labels.numpy().astype(np.int32).reshape((-1,))
                train_h5.append('indices', id_np)
                train_h5.append('labels', labels_np)

            # val
            for batch_idx, (inputs, labels) in enumerate(tqdm(val_loader, desc=f"val_rank{rank}")):
                inputs = inputs.to(device)
                _, _, id_b, _ = base_model.encode(inputs)
                id_np = id_b.cpu().numpy().astype(np.int32)
                labels_np = labels.numpy().astype(np.int32).reshape((-1,))
                val_h5.append('indices', id_np)
                val_h5.append('labels', labels_np)

        train_h5.close()
        val_h5.close()
    except Exception as e: 
        if rank == 0: 
            mlflow.log_text(str(e), "error.txt") 
            raise
    finally:
        if rank == 0 and run is not None:
            mlflow.end_run()

if __name__ == '__main__':
    main()
