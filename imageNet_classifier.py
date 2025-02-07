import torch, json, urllib.request, os
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import datasets, transforms
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.optim as optim
import numpy as np

class CustomImageNetDataset(Dataset):
    def __init__(self, root, transform, class_to_imagenet_idx):
        self.dataset = datasets.ImageFolder(root=root, transform=transform)
        
        # Overwrite class_to_idx with ImageNet indices
        self.class_to_idx = class_to_imagenet_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}  # Reverse mapping

        # Remap targets to ImageNet indices
        self.targets = [self.class_to_idx[self.dataset.classes[label]] for label in self.dataset.targets]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        label = self.targets[idx]  # Get correctly mapped label
        return image, label

# Define ImageNet class mapping


class_index_url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
with urllib.request.urlopen(class_index_url) as url:
    class_idx = json.loads(url.read().decode())

idx_to_class = {int(key): value[1] for key, value in class_idx.items()}  # Map index to class name
class_to_idx = {value[1]: int(key) for key, value in class_idx.items()}  # Map class name to index

# Define transformation (ResNet-style transformation will be done later in the code by calling weights.transform())
transform = transforms.Compose([
    # transforms.Resize((80,80)),
    transforms.ToTensor(),
])

# Dataset directory
data_dir = "/home/abghamtm/work/masking_comparison/image/100class-vqvae-reconstruction"

# Get folder names from dataset
imagenet_labels = os.listdir(data_dir)  # Ensure folder names match ImageNet labels

# Filter only matching classes
custom_to_imagenet_idx = {cls_name : class_to_idx[cls_name] for cls_name in imagenet_labels if cls_name in class_to_idx}

# Load custom dataset
dataset = CustomImageNetDataset(root=data_dir, transform=transform, class_to_imagenet_idx=custom_to_imagenet_idx)

batch_size = 256
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=12)

weights = ResNet50_Weights.IMAGENET1K_V2
preprocess = weights.transforms()
torch.cuda.set_device(1)  # Use GPU 1 (if desired)
torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
# num_ftrs = model.fc.in_features # Modify the final layer to match your number of classes
# model.fc = nn.Linear(num_ftrs, 100) # Modify the final layer to match your number of classes
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
num_epochs = 30  # Adjust as needed

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in tqdm(dataloader):
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
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = 100 * correct / total

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
    torch.save(model.state_dict(), f"/home/abghamtm/work/masking_comparison/checkpoint/classifier/resnet50/weights_epoch{str(epoch + 1).zfill(2)}.pth")


# # Define classifier and load saved model(weights)
# classifier = resnet50(pretrained=False)
# num_ftrs = classifier.fc.in_features
# classifier.fc = nn.Linear(num_ftrs, 100)
# models_list = os.listdir("/home/abghamtm/work/masking_comparison/checkpoint/classifier/resnet50/")
# models_list.sort()
# for model_name in models_list:
#     print(model_name)
#     classifier.load_state_dict(torch.load(os.path.join("/home/abghamtm/work/masking_comparison/checkpoint/classifier/resnet50/",model_name)))
#     classifier.to(device)
#     classifier.eval()  # Set model to evaluation mode
#     correct = 0
#     total = 0
#     total_loss = 0

#     with torch.no_grad():
#         for inputs, labels in tqdm(test_loader):
#             inputs = preprocess(inputs)
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = classifier(inputs)
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#             loss = criterion(outputs, labels)
#             total_loss += loss.item()

#     accuracy = correct / total
#     print(f'Accuracy: {accuracy * 100:.2f}%')
#     print(f'loss: {total_loss:.2f}%')
#     # return accuracy, error_rate


