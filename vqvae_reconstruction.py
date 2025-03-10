import argparse, random, sys, os, torch, json, urllib.request
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms, utils 

import numpy as np
from tqdm import tqdm

from vqvae import FlatVQVAE
# from scheduler import CycleScheduler
import distributed as dist
import neptune.new as neptune

os. nice (19)

def rand_img_selection(dataset, images_per_class=100, seed=42):
    '''
    Randomly select fixed number of images from each folder (classes)
    '''
    # Fix the random seed for reproducability
    random.seed(seed)

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

class CustomImageDataset(Dataset):
    def __init__(self, subset):
        self.subset = subset
        self.classes = subset.classes  # Keep mapped class names
        self.class_to_idx = subset.class_to_idx  # Keep class-to-index mapping
        # self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}  # Reverse mapping

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]  # Get image and numerical label
        class_name = self.classes[label]  # Convert label to class name
        class_label = self.class_to_idx[class_name] # Convert
        return img, class_label

def main(args):
    torch.cuda.set_device(3)  # Use GPU 1 (if desired)
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

    # Change wnid folder name to corresponding indices and names
    ## Call essential library to map indices to proper names and wnid
    class_index_url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    with urllib.request.urlopen(class_index_url) as url:
        class_idx = json.loads(url.read().decode())
    idx_to_class = {int(key): value[1] for key, value in class_idx.items()}
    wnid_to_idx = {value[0]: int(key) for key, value in class_idx.items()}

    ## Correct classes and class_to_idx attribute and replace wnid constituting classes and key in class_to_idx with true class names and values of class_to_idx which automatically sets from 0 to 99 to true labels
    dataset_class_to_idx = {wnid: wnid_to_idx[wnid] for wnid in dataset.classes if wnid in wnid_to_idx}
    dataset.class_to_idx = dataset_class_to_idx
    for i, cls in enumerate(dataset.classes):
        dataset.classes[i] = idx_to_class[dataset.class_to_idx[cls]]
        class_to_idx = {class_idx[str(value)][1]: value for value in dataset.class_to_idx.values()}
    dataset.class_to_idx = class_to_idx

    # Select a sample from all images
    selected_dataset = rand_img_selection(dataset)
    # Change the label to true indices by a custom dataset
    wrapped_dataset = CustomImageDataset(selected_dataset)
    data_loader = DataLoader(wrapped_dataset, batch_size=256 // args.n_gpu, shuffle=True, num_workers=12)


    model_vqvae = FlatVQVAE().to(device)
    model_vqvae.load_state_dict(torch.load(args.ckpt_vqvae, map_location=device))
    model_vqvae = model_vqvae.to(device)
    model_vqvae.eval()
    epoch = args.epoch

    if args.distributed:
        model_vqvae = nn.parallel.DistributedDataParallel(
            model_vqvae,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )

    # Save the final model
    os.makedirs(args.save_path_models, exist_ok=True)
    if dist.is_primary():
        model_vqvae.eval()
        all_indices = []
        all_quantizes = []
        all_labels = []
        with torch.no_grad():
            for j, (images, labels) in tqdm(enumerate(data_loader), desc='Recall trained model to save codebook indices', leave=False): # Ignore the label in DataLoader
                images = images.float().to(device)
                quant_b, _, id_b, _, _ = model_vqvae.encode(images)
                outputs = model_vqvae.decode(quant_b)
                for idx, (label, out) in enumerate(zip(labels, outputs)):
                    print(label)
                    # class_name = selected_dataset.classes[labels[idx].item()]
                    class_folder = os.path.join(args.save_path_imgs, str(label.item()))
                    os.makedirs(class_folder, exist_ok=True)

                    save_file = os.path.join(class_folder, f"reconstructed_{label.item()}_{j * data_loader.batch_size + idx + 1:05d}.png")
                    utils.save_image(
                        torch.cat([out.unsqueeze(0)], 0),
                        save_file,
                        nrow=2,
                        normalize=True,
                        range=(-1, 1),
                    )

                all_indices.append(id_b.cpu())
                all_quantizes.append(quant_b.cpu())
                all_labels.extend(labels.cpu().numpy())
                

        # Concatenate all indices into a single tensor and save it
        indices_tensor = torch.cat(all_indices, dim=0)
        quantizes_tensor = torch.cat(all_quantizes, dim=0)
        all_labels = np.array(all_labels)

        indices_path = os.path.join(args.save_path_models, f"indices_epoch{epoch}_flat_vqvae80x80_144x456codebook.npy")
        quantized_path = os.path.join(args.save_path_models, f"quantized_epoch{epoch}_flat_vqvae80x80_144x456codebook.npy")
        np.save(os.path.join(args.save_path_models, f'labels_epoch{epoch}_flat_vqvae80x80_144x456codebook.npy'), all_labels)

        np.save(indices_path, indices_tensor.numpy())
        np.save(quantized_path, quantizes_tensor.numpy())

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
    parser.add_argument("--epoch", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--sched", type=str)
    parser.add_argument('--ckpt_vqvae', type=str, default="/home/abghamtm/work/masking_comparison/checkpoint/vqvae/model_epoch80_flat_vqvae80x80_144x456codebook.pth")

    # parser.add_argument("path", type=str)

    args = parser.parse_args()

    print(args)

    dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))
