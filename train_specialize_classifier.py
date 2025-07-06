import torch, os, argparse, re
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.optim as optim
import numpy as np
from vqvae import FlatVQVAE
import distributed as dist

class CustomMaskingDataset(Dataset):
    def __init__(self, scr_img_pth, masked_img_pth, masking_type, max_percentage=30, transform=None):
        assert masking_type in ['random', 'additive', 'selective'], "masking_type must be 'random', 'additive', or 'selective'"
        self.scr_img_pth = scr_img_pth
        self.masked_img_pth = masked_img_pth
        self.masking_type = masking_type
        self.max_percentage = max_percentage
        self.transform = transform
        self.samples = self._collect_samples()

    def _collect_samples(self):
        source_files = os.listdir(self.scr_img_pth)
        masked_files = os.listdir(self.masked_img_pth)
        orig_pattern = re.compile(r'(\d{5})_OriginalImage_TrueLabel=(\d+)\.npy')
        mask_pattern = re.compile(
            r'(\d{5})_MaskPercentage=(\d+)_'
            f'{"Random" if self.masking_type=="random" else "Additive" if self.masking_type=="additive" else "Selective"}'
            r'MaskingWithTransformer_TrueLabel=(\d+)\.npy'
        )
        samples = []
        for fname in source_files:
            m = orig_pattern.fullmatch(fname)
            if m:
                idx, label = m.groups()
                samples.append(('scr', fname, int(label)))
        for fname in masked_files:
            m = mask_pattern.fullmatch(fname)
            if m:
                idx, percentage, label = m.groups()
                percentage = int(percentage)
                if percentage <= self.max_percentage:
                    samples.append(('masked', fname, int(label)))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dir_type, fname, label = self.samples[idx]
        if dir_type == 'scr':
            full_path = os.path.join(self.scr_img_pth, fname)
        else:
            full_path = os.path.join(self.masked_img_pth, fname)
        img = np.load(full_path)
        if self.transform:
            img = self.transform(img)
        img = torch.from_numpy(img) if not torch.is_tensor(img) else img
        if img.ndim == 3 and img.shape[-1] == 3:
            img = img.permute(2, 0, 1)
        return img, label

    @property
    def labels(self):
        return [label for _, _, label in self.samples]


def data_loader(scr_img_pth, masked_img_pth, masking_type):
    dataset = CustomMaskingDataset(scr_img_pth=scr_img_pth, masked_img_pth=masked_img_pth, masking_type=masking_type)
    labels = dataset.labels
    train_idx, test_idx = train_test_split(
        np.arange(len(labels)),
        test_size=0.2,
        stratify=labels,
        random_state=42
    )
    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)
    return train_loader, test_loader

# Initialize CUDA
def setup_resources():
    torch.cuda.init()
    torch.cuda.set_device(3)
    torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device

def train_model(device, train_loader, model_masking_type, model_path='/home/abghamtm/work/masking_comparison/checkpoint/classifier/resnet50/'):
    assert model_masking_type in ['random', 'additive', 'selective'], "masking_type must be 'random', 'additive', or 'selective'"
    model_full_path = os.path.join(model_path, f'{model_masking_type}/')
    # Initialize the model
    weights = ResNet50_Weights.IMAGENET1K_V2
    preprocess = weights.transforms()
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Training loop
    num_epochs = 20  # Adjust as needed

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in tqdm(train_loader):
            inputs = preprocess(inputs)
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Accuracy calculation
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Calculate average loss and accuracy
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
        torch.save(model.state_dict(), f"{model_full_path}weights_epoch{str(epoch + 1).zfill(2)}.pth")

def test_trained_model(device, test_loader, model_masking_type, model_path='/home/abghamtm/work/masking_comparison/checkpoint/classifier/resnet50/'):
    assert model_masking_type in ['random', 'additive', 'selective'], "masking_type must be 'random', 'additive', or 'selective'"
    model_full_path = os.path.join(model_path, f'{model_masking_type}/')
    
    # Initialize the model
    weights = ResNet50_Weights.IMAGENET1K_V2
    preprocess = weights.transforms()
    classifier = resnet50(pretrained=False)
    models_list = os.listdir(model_full_path)
    models_list.sort()
    criterion = nn.CrossEntropyLoss()
    for model_full_name in models_list:
        print(model_full_name)
        classifier.load_state_dict(torch.load(os.path.join(model_full_path, model_full_name)))
        classifier.to(device)
        classifier.eval()  # Set model to evaluation mode
        correct = 0
        total = 0
        total_loss = 0

        with torch.no_grad():
            for inputs, labels in tqdm(test_loader):
                inputs = preprocess(inputs)
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = classifier(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss = criterion(outputs, labels)
                total_loss += loss.item()

        accuracy = correct / total
        print(f'Accuracy: {accuracy * 100:.2f}%')
        print(f'loss: {total_loss:.2f}')

# def model_inference(loader, model_name, model_masking_suffix):
#     assert model_masking_suffix in ['random', 'additive', 'selective'], "masking_type must be 'random', 'additive', or 'selective'"
#     model_full_name = model_name+'_'+model_masking_suffix+'.pth'

#     # Initialize the model
#     weights = ResNet50_Weights.IMAGENET1K_V2
#     preprocess = weights.transforms()
#     classifier = resnet50(pretrained=False)
#     classifier.load_state_dict(torch.load(f'/home/abghamtm/work/masking_comparison/checkpoint/classifier/resnet50/{model_full_name}'))
#     classifier.to(device)
#     classifier.eval()

def main():
    device = setup_resources()
    
    add_tr_loader, add_te_loader = data_loader(scr_img_pth='/home/abghamtm/work/masking_comparison/image/recons/original_img_reconstrcted_data/', masked_img_pth='/home/abghamtm/work/masking_comparison/image/recons/aditive_img_data/', masking_type='additive')
    for inputs, labels in add_tr_loader:
        print(inputs.shape, labels.shape)
        break
    train_model(device, train_loader=add_tr_loader, model_masking_type='additive')
    test_trained_model(device, test_loader=add_te_loader, model_masking_type='additive')

    rand_tr_loader, rand_te_loader = data_loader(scr_img_pth='/home/abghamtm/work/masking_comparison/image/recons/original_img_reconstrcted_data/', masked_img_pth='/home/abghamtm/work/masking_comparison/image/recons/random_img_data/', masking_type='random')
    for inputs, labels in rand_tr_loader:
        print(inputs.shape, labels.shape)
        break
    train_model(device, train_loader=rand_tr_loader, model_masking_type='random')
    test_trained_model(device, test_loader=rand_te_loader, model_masking_type='random')

    select_tr_loader, select_te_loader = data_loader(scr_img_pth='/home/abghamtm/work/masking_comparison/image/recons/original_img_reconstrcted_data/', masked_img_pth='/home/abghamtm/work/masking_comparison/image/recons/selective_img_data/', masking_type='selective')
    for inputs, labels in select_tr_loader:
        print(inputs.shape, len(labels))
        break
    train_model(device, train_loader=select_tr_loader, model_masking_type='selective')
    test_trained_model(device, test_loader=select_te_loader, model_masking_type='selective')

if __name__ == "__main__":
    main()