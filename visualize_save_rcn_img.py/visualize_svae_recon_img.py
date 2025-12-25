# decode_samples.py
import argparse, os, random, h5py, numpy as np, torch
from PIL import Image
from vqvae import FlatVQVAE

def unnormalize(img_tensor):
    # img_tensor: torch.Tensor (C,H,W) in ImageNet-normalized space
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1,1,1)
    img = img_tensor * std + mean
    img = img.clamp(0.0, 1.0)
    return (img * 255).byte().permute(1,2,0).cpu().numpy()

def load_vqvae(ckpt_path, device):
    model = FlatVQVAE().to(device)
    sd = torch.load(ckpt_path, map_location=device)
    # sd might already be a state_dict
    try:
        model.load_state_dict(sd, strict=False)
    except Exception:
        # try to strip/add module prefix
        new = {}
        for k,v in sd.items():
            nk = k.replace('module.', '') if k.startswith('module.') else 'module.' + k
            new[nk] = v
        model.load_state_dict(new, strict=False)
    model.eval()
    return model

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--h5', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--n', type=int, default=16)
    p.add_argument('--out', default='decodes')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = load_vqvae(args.ckpt, device)

    with h5py.File(args.h5, 'r') as f:
        if 'indices' not in f or 'labels' not in f:
            raise RuntimeError("HDF5 must contain 'indices' and 'labels' datasets.")
        total = f['indices'].shape[0]
        indices = random.sample(range(total), min(args.n, total))
        for idx in indices:
            ids = f['indices'][idx]   # shape (H, W)
            label = int(f['labels'][idx])
            ids_t = torch.from_numpy(ids).unsqueeze(0).long().to(device)  # (1,H,W)
            with torch.no_grad():
                recon = model.decode_code(ids_t)  # (1, C, H, W)
            recon = recon.squeeze(0).cpu()  # (C,H,W)
            img = unnormalize(recon)
            Image.fromarray(img).save(os.path.join(args.out, f"sample_{idx}_label_{label}.png"))
            print("Wrote sample", idx, "label", label)

if __name__ == '__main__':
    main()
