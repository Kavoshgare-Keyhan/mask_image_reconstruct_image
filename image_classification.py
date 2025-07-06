import torch, os, argparse, re, shutil
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.optim as optim
import numpy as np
import neptune.new as neptune

run = neptune.init_run(
    project="tns/Vqvae-transformer",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwODg4OTU0Yy0xODAyLTRiM2QtYjYzYi0xMWQxYThmYWJlOWQifQ==",
    capture_stdout = False,
    capture_stderr = False,
    # with_id="MAS-389"
)

class CustomMaskingDataset(Dataset):
    def __init__(self, img_pth, masking_type, mask_percentage, transform=None):
        assert masking_type in ['random', 'additive', 'selective'], "masking_type must be 'random', 'additive', or 'selective'"
        self.img_pth = img_pth
        self.masking_type = masking_type
        self.mask_percentage = mask_percentage
        self.transform = transform
        self.samples = self._collect_samples()

    def _collect_samples(self):
        files = os.listdir(self.img_pth)
        file_pattern = re.compile(
            r'(\d{5})_MaskPercentage=(\d+)_'
            f'{"Random" if self.masking_type=="random" else "Additive" if self.masking_type=="additive" else "Selective"}'
            r'.*_TrueLabel=(\d+)\.npy'
        )
        samples = []
        for fname in files:
            m = file_pattern.fullmatch(fname)
            if m:
                idx, percentage, label = m.groups()
                percentage = int(percentage)
                if percentage == self.mask_percentage:
                    samples.append(('masked', fname, int(label)))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dir_type, fname, label = self.samples[idx]
        if dir_type == 'masked':
            full_path = os.path.join(self.img_pth, fname)
        img = np.load(full_path)
        if self.transform:
            img = self.transform(img)
        img = torch.from_numpy(img) if not torch.is_tensor(img) else img
        if img.ndim == 3 and img.shape[-1] == 3:
            img = img.permute(2, 0, 1)
        return img, label, fname, full_path

    @property
    def labels(self):
        return [label for _, _, label in self.samples]


def data_loader(img_pth, masking_type, mask_percentage):
    dataset = CustomMaskingDataset(img_pth=img_pth, masking_type=masking_type, mask_percentage=mask_percentage)
    labels = dataset.labels
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    return dataloader

# Initialize CUDA
def setup_resources():
    torch.cuda.init()
    torch.cuda.set_device(3)
    torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device

def classify(device, dataloader, model_masking_type, model_path='/home/abghamtm/work/masking_comparison/checkpoint/classifier/resnet50/', model_name='weights_epoch20.pth', pred_path='/home/abghamtm/work/masking_comparison/image/recons/prediction/'):
    assert model_masking_type in ['random', 'additive', 'selective'], "masking_type must be 'random', 'additive', or 'selective'"
    model_full_path = os.path.join(model_path, f'{model_masking_type}/')
    prediction_full_path = os.path.join(pred_path, f'{model_masking_type}/')
    
    # Initialize the model
    weights = ResNet50_Weights.IMAGENET1K_V2
    preprocess = weights.transforms()
    classifier = resnet50(pretrained=False)
    classifier.load_state_dict(torch.load(os.path.join(model_full_path, model_name)))
    classifier.to(device)
    classifier.eval()  # Set model to evaluation mode


    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    total_loss = 0

    with torch.no_grad():
        for inputs, labels, fnames in tqdm(dataloader):
            inputs = preprocess(inputs)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = classifier(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            # Save predictions
            # Convert tensor to numpy array with shape [c, w, h]
            for i in range(inputs.size(0)):
                original_fname = fnames[i] if isinstance(fnames, list) else fnames
                orig_path = orig_paths[i] if isinstance(orig_paths, list) else orig_paths
                pred_label = predicted[i].item()
                # Remove .npy and add new suffix *********************************
                base, ext = os.path.splitext(original_fname)
                new_fname = f"{base}_PredictionLabel={pred_label}{ext}"
                save_path = os.path.join(prediction_full_path, new_fname)
                # Copy the original file
                import shutil
                shutil.copy(orig_path, save_path)
    accuracy = correct / total
    classification_err = 1-accuracy
    run[f"recons/average_classification_error_{model_masking_type}"].log(classification_err)
    run[f"recons/average_cross_entropy_error_{model_masking_type}"].log(total_loss/total)

def main():
    device = setup_resources()
    mask_percentages = np.arange(0.1, 1.1, 0.1)
    mask_percentages = np.append(mask_percentages, [.85,.95])
    mask_percentages = np.sort(mask_percentages)
    mask_percentages *= 100  # Convert to percentage

    for perc in mask_percentages:
        add_loader = data_loader(img_pth='/home/abghamtm/work/masking_comparison/image/recons/aditive_img_data/', masking_type='additive', mask_percentage=perc)
        classify(device, dataloader=add_loader, model_masking_type='additive')

        rand_loader = data_loader(img_pth='/home/abghamtm/work/masking_comparison/image/recons/random_img_data/', masking_type='random', mask_percentage=perc)
        classify(device, dataloader=rand_loader, model_masking_type='random')

        select_loader = data_loader(img_pth='/home/abghamtm/work/masking_comparison/image/recons/selective_img_data/', masking_type='selective', mask_percentage=perc)
        classify(device, dataloader=select_loader, model_masking_type='selective')

if __name__ == "__main__":
    main()
