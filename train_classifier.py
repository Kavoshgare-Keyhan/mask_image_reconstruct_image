# 📦 Imports
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import optuna
import mlflow
import mlflow.pytorch

# 🖥️ Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 📁 Dataset paths
data_dir = "path/to/your/dataset"  # Must contain train/, val/, test/
num_classes = 1000  # Adjust based on your dataset

# 🧪 Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 📂 Load datasets
train_set = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
val_set = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=transform)
test_set = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform)

# Add Global Tracker
best_model_acc = 0.0
best_model_state = None

# 🧠 Optuna objective function
def objective(trial):
    # 🔧 Suggest hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    num_epochs = trial.suggest_int('num_epochs', 5, 15)

    # 🔄 Data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # 🧠 Load and modify pretrained ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    # ⚙️ Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)

    # 📁 MLflow tracking
    mlflow.start_run(run_name=f"Optuna_Trial_{trial.number}")
    mlflow.log_params({
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "num_epochs": num_epochs
    })

    # 🏋️ Training loop
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # 🔍 Validation
        model.eval()
        val_preds, val_labels = [], []
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = outputs.max(1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_labels)
        val_acc = accuracy_score(val_labels, val_preds)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_acc, step=epoch)

    # Track the best model
    global best_model_acc, best_model_state
    if val_acc > best_model_acc:
        best_model_acc = val_acc
        best_model_state = model.state_dict()

    # 📊 Confusion matrix
    cm = confusion_matrix(val_labels, val_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_set.classes)
    disp.plot(cmap='Blues', xticks_rotation='vertical')
    plt.title("Validation Confusion Matrix")
    cm_path = f"confusion_matrix_trial_{trial.number}.png"
    plt.savefig(cm_path)
    plt.close()

    # 📁 Log metrics and model
    mlflow.log_artifact(cm_path)
    mlflow.pytorch.log_model(model, "model")
    mlflow.end_run()

    return val_acc

# 🚀 Run Optuna study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

# Save best model weights
torch.save(best_model_state, "best_model.pth")
print("✅ Best model saved to best_model.pth")

# 🏆 Best hyperparameters
print("Best hyperparameters found:")
for key, value in study.best_params.items():
    print(f"{key}: {value}")

# 🔧 Load best hyperparameters from Optuna
best_params = study.best_params
lr = best_params['lr']
weight_decay = best_params['weight_decay']
batch_size = best_params['batch_size']
num_epochs = best_params['num_epochs']

# 🔄 Combine train + val
full_train_set = ConcatDataset([train_set, val_set])
train_loader = DataLoader(full_train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# 📁 Directory to save models
os.makedirs("saved_models", exist_ok=True)

# 🧠 Track best model
best_test_acc = 0.0
best_model_path = None

# 🏋️ Retrain and evaluate multiple times
for run_id in range(3):
    print(f"\n🔁 Retraining model #{run_id + 1}")
    mlflow.start_run(run_name=f"Retrain_Run_{run_id + 1}")

    mlflow.log_params({
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "run_id": run_id + 1
    })

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load("best_model.pth"))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        scheduler.step(train_loss)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_accuracy", train_acc, step=epoch)

    model_path = f"saved_models/model_run_{run_id + 1}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model #{run_id + 1} saved to {model_path}")
    mlflow.pytorch.log_model(model, "model")

    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    test_acc = accuracy_score(test_labels, test_preds)
    print(f"🎯 Model #{run_id + 1} Test Accuracy: {test_acc:.4f}")
    mlflow.log_metric("test_accuracy", test_acc)

    cm = confusion_matrix(test_labels, test_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_set.classes)
    disp.plot(cmap='Blues', xticks_rotation='vertical') 
    plt.title("Best Model Test Confusion Matrix") 
    plt.savefig("best_model_confusion_matrix.png") 
    plt.show()