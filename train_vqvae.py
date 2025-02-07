import argparse, random, sys, os, torch, json, urllib.request
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, Subset

from torchvision import datasets, transforms, utils 
from torchvision.datasets import ImageNet

import numpy as np
from tqdm import tqdm

from vqvae import FlatVQVAE
# from scheduler import CycleScheduler
from torch.optim.lr_scheduler import CyclicLR
import distributed as dist

from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import neptune.new as neptune

os. nice (19)

# run = neptune.init_run(
#     project="tns/Vqvae-transformer",
#     api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwODg4OTU0Yy0xODAyLTRiM2QtYjYzYi0xMWQxYThmYWJlOWQifQ==",
#     capture_stdout=False,
#     capture_stderr=False,
# )
def select_images_per_class(dataset, images_per_class=100, seed=42):
    random.seed(seed)  # Set random seed for reproducibility

    class_indices = {class_name: [] for class_name in dataset.class_to_idx}
    for idx, (_, label) in enumerate(dataset.samples):
        class_name = dataset.classes[label]
        class_indices[class_name].append(idx)

    selected_indices = []
    for class_name, indices in class_indices.items():
        selected_indices.extend(random.sample(indices, min(images_per_class, len(indices))))

    subset = Subset(dataset, selected_indices)

    # Assign the same attributes as the original dataset
    subset.classes = dataset.classes
    subset.class_to_idx = dataset.class_to_idx

    return subset

class DatasetWithClassLabels(Dataset):
    def __init__(self, subset):
        self.subset = subset
        self.classes = subset.classes  # Keep mapped class names
        self.class_to_idx = subset.class_to_idx  # Keep class-to-index mapping
        # self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}  # Reverse mapping

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label_idx = self.subset[idx]  # Get image and numerical label
        class_name = self.classes[label_idx]  # Convert label to class name
        # class_label = self.class_to_idx[class_name] # Convert
        return img, class_name

def train(epoch, loader, model, optimizer, scheduler, device):
    if dist.is_primary():
        loader = tqdm(loader)

    criterion = nn.MSELoss()

    latent_loss_weight = 0.35
    diversity_loss_weight = 0.0001

    mse_sum = 0
    mse_n = 0

    for i, (img, label) in enumerate(loader):
        model.zero_grad()
        img = img.to(device)
        out, latent_loss, diversity_loss, _ = model(img)
        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss + diversity_loss_weight * diversity_loss
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        part_mse_sum = recon_loss.item() * img.shape[0]
        part_mse_n = img.shape[0]
        comm = {"mse_sum": part_mse_sum, "mse_n": part_mse_n}
        comm = dist.all_gather(comm)

        for part in comm:
            mse_sum += part["mse_sum"]
            mse_n += part["mse_n"]

        if dist.is_primary():
            lr = optimizer.param_groups[0]["lr"]

            loader.set_description(
                (
                    f"epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; "
                    f"latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; "
                    f"lr: {lr:.5f}"
                )
            )


def main(args):
    torch.cuda.set_device(1)  # Use GPU 1 (if desired)
    torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.distributed = dist.get_world_size() > 1

    transform = transforms.Compose(
        [
            transforms.Resize((80,80)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    dataset = datasets.ImageFolder("/local/reyhasjb/datasets/Imagenet-100class/train",transform=transform)

    # Set true labels for datasets with folder name labels 
    class_index_url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    with urllib.request.urlopen(class_index_url) as url:
        class_idx = json.loads(url.read().decode())

    idx_to_class = {int(key): value[1] for key, value in class_idx.items()}
    wnid_to_idx = {value[0]: int(key) for key, value in class_idx.items()}
    dataset_class_to_idx = {wnid: wnid_to_idx[wnid] for wnid in dataset.classes if wnid in wnid_to_idx}
    dataset.class_to_idx = dataset_class_to_idx
    for i, cls in enumerate(dataset.classes):
        dataset.classes[i] = idx_to_class[dataset.class_to_idx[cls]]
        class_to_idx = {class_idx[str(value)][1]: value for value in dataset.class_to_idx.values()}
    dataset.class_to_idx = class_to_idx

    selected_dataset = select_images_per_class(dataset)
    wrapped_dataset = DatasetWithClassLabels(selected_dataset)

    sampler = dist.data_sampler(wrapped_dataset, shuffle=True, distributed=args.distributed)
    loader = DataLoader(wrapped_dataset, batch_size=256 // args.n_gpu, sampler=sampler, num_workers=12)


    model = FlatVQVAE().to(device)


    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # run["train/lr"].log(args.lr)
    scheduler = None
    if args.sched == "cycle":
        scheduler = CyclicLR(
        optimizer, 
        base_lr=args.lr * 0.1, 
        max_lr=args.lr, 
        step_size_up=len(loader) * args.epoch * 0.05, 
        mode="triangular",
        cycle_momentum=False)
    
    for i in range(args.epoch):
        train(i, loader, model, optimizer, scheduler, device)
        # if i==99: torch.save(model.state_dict(), f"/home/abghamtm/work/masking_comparison/checkpoint/vqvae/test_model_epoch{i+1}_flat_vqvae80x80_144x456codebook.pth")

    # Save the final model
    os.makedirs(args.save_path_models, exist_ok=True)
    if dist.is_primary():
        model.eval()
        all_indices = []
        all_quantizes = []
        with torch.no_grad():
            for j, (images, labels) in tqdm(enumerate(loader), desc='Recall trained model to save codebook indices', leave=False): # Ignore the label in DataLoader
                images = images.float().to(device)
                quant_b, _, id_b, _, _ = model.encode(images)
                outputs, _, _, _ = model(images)
                for idx, (label, out) in enumerate(zip(labels, outputs)):
                    print(label)
                    # class_name = selected_dataset.classes[labels[idx].item()]
                    class_folder = os.path.join(args.save_path_imgs, label)
                    os.makedirs(class_folder, exist_ok=True)

                    save_file = os.path.join(class_folder, f"reconstructed_{label}_{j * loader.batch_size + idx + 1:05d}.png")
                    utils.save_image(
                        torch.cat([out.unsqueeze(0)], 0),
                        save_file,
                        nrow=2,
                        normalize=True,
                        range=(-1, 1),
                    )

                all_indices.append(id_b.cpu())
                all_quantizes.append(quant_b.cpu())

        # Concatenate all indices into a single tensor and save it
        indices_tensor = torch.cat(all_indices, dim=0)
        quantizes_tensor = torch.cat(all_quantizes, dim=0)

        indices_path = os.path.join(args.save_path_models, f"indices_epoch{i+1}_flat_vqvae80x80_144x456codebook.npy")
        quantized_path = os.path.join(args.save_path_models, f"quantized_epoch{i+1}_flat_vqvae80x80_144x456codebook.npy")
        model_path = os.path.join(args.save_path_models, f"model_epoch{i+1}_flat_vqvae80x80_144x456codebook.pth")

        np.save(indices_path, indices_tensor.numpy())
        np.save(quantized_path, quantizes_tensor.numpy())
        torch.save(model.state_dict(), model_path)
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)

    port = (
        2 ** 15
        + 2 ** 14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )
    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")
    parser.add_argument("--save_path_models", default="/home/abghamtm/work/masking_comparison/checkpoint/vqvae/")
    parser.add_argument("--save_path_imgs", default="/home/abghamtm/work/masking_comparison/image/100class-vqvae-reconstruction/")
    parser.add_argument("--size", type=int, default=80)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--sched", type=str)
    # parser.add_argument('--ckpt_vqvae', type=str, default="checkpoint/flat_vqvae_80x80_144x456codebook_100class_051.pt")

    # parser.add_argument("path", type=str)

    args = parser.parse_args()

    print(args)

    dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))
