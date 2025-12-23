# train_vqvae.py
import os, yaml, argparse, torch, mlflow, subprocess
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from vqvae import FlatVQVAE
from data_utils import CustomImageNetDataV2, CustomLoader
from gpu_utils import select_gpus
import torch.distributed as dist
from tqdm import tqdm

os.nice(19)

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

def save_checkpoint(model, path):
    # Save the state_dict of the underlying module (not the DDP wrapper)
    state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save(state, path)

def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.gpu_ids is not None: cfg['multiprocessing']['gpu']['gpu_ids'] = args.gpu_ids
    selected_gpu_ids = select_gpus(cfg['multiprocessing']['gpu'])
    rank, world_size, local_rank = setup_distributed(selected_gpu_ids)
    device = torch.device(f"cuda:{selected_gpu_ids[local_rank]}")

    path_cfg = cfg['path']
    model_dir = path_cfg['model']['vqvae']
    os.makedirs(model_dir, exist_ok=True)

    params = cfg['params']['vqvae']
    batch_size = int(params['batch_size'])
    lr = float(params.get('lr', 1e-4))
    weight_decay = float(params.get('weight_decay', 0.0))
    num_epochs = int(params.get('num_epochs', 10))
    latent_loss_weight = float(params.get('latent_loss_weight', 1.0))
    diversity_loss_weight = float(params.get('diversity_loss_weight', 0.0))

    # prepare datasets
    train_set = CustomImageNetDataV2(image_dir=path_cfg['image_net_train_local'], image_type='original', folder_label='word_net_id')
    val_set = CustomImageNetDataV2(image_dir=path_cfg['image_net_val_local'], image_type='original', folder_label='word_net_id')

    train_loader = CustomLoader(train_set, batch_size=batch_size, threads=cfg.get('train', {}).get('num_workers', 4),
                                shuffle=True, distributed=dist.is_initialized(), world_size=world_size, rank=rank,
                                pin_memory=cfg.get('train', {}).get('pin_memory' ,False)).data_loader
    val_loader = CustomLoader(val_set, batch_size=batch_size, threads=cfg.get('train', {}).get('num_workers', 4),
                              shuffle=False, distributed=dist.is_initialized(), world_size=world_size, rank=rank,
                              pin_memory=cfg.get('train', {}).get('pin_memory', False)).data_loader

    model = FlatVQVAE().to(device)
    if dist.is_initialized():
        model = DDP(model, device_ids=[device.index])

    recon_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    try:
        # MLflow start
        if rank == 0:
            mlflow.set_experiment("vqvae_experiment")
            commit = None
            try:
                commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()
            except Exception:
                pass
            run = mlflow.start_run(run_name="vqvae_training", tags={"git_commit": commit})
            print("Experiment ID:", run.info.experiment_id) 
            print("Run ID:", run.info.run_id)
            mlflow.log_params({
                "batch_size": batch_size, "lr": lr, "weight_decay": weight_decay,
                "num_epochs": num_epochs, "latent_loss_weight": latent_loss_weight,
                "diversity_loss_weight": diversity_loss_weight
            })
        else:
            print(f"rank is {rank}, no mlflow logging")
            run = None
        # ---- training loop ----
        best_val = float('inf')
        best_epoch = -1
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            for inputs, _ in tqdm(train_loader):
                inputs = inputs.to(device)
                optimizer.zero_grad()
                recon, latent_diff, ids, num_used = model(inputs)
                recon_loss = recon_criterion(recon, inputs)
                # latent_diff in encode returned as scalar tensor per batch; ensure correct weight
                loss = recon_loss + latent_loss_weight * latent_diff
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)

            train_loss /= len(train_loader.dataset)
            # validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, _ in tqdm(val_loader):
                    inputs = inputs.to(device)
                    recon, latent_diff, ids, num_used = model(inputs)
                    recon_loss = recon_criterion(recon, inputs)
                    loss = recon_loss + latent_loss_weight * latent_diff
                    val_loss += loss.item() * inputs.size(0)
            val_loss /= len(val_loader.dataset)

            if rank == 0:
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                print(f"[Epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

            # save checkpoint per epoch and track best
            ckpt_path = os.path.join(model_dir, f"model_epoch_{epoch}_vqvae.pth")
            save_checkpoint(model, ckpt_path)
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                best_model_path = os.path.join(model_dir, f"best_vqvae_model.pth")
                save_checkpoint(model, best_model_path)

        if rank == 0:
            mlflow.log_param("best_epoch", best_epoch)
            mlflow.log_artifact(best_model_path, artifact_path="checkpoints")
            
    except Exception as e: 
        if rank == 0: 
            mlflow.log_text(str(e), "error.txt") 
            raise
    finally:
        if rank == 0 and run is not None:
            mlflow.end_run()

if __name__ == "__main__":
    try:
        main()
    except Exception as e: 
        # if rank == 0: 
        mlflow.log_text(str(e), "error.txt") 
        raise
