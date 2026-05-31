#----------------------------------------------#
#       Import packages
#----------------------------------------------#
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import mlflow
import torchvision


#----------------------------------------------#
#      Configuration and Data loading
#----------------------------------------------#
# config = {
#     "dataset": "Fashion-MNIST",
#     "model": "FashionCNN",
#     "batch_size": 64,
#     "learning_rate": 0.001,
#     "optimizer": "Adam",
#     "epochs": 5,
#     "dropout": 0.3
# }
config = {
    "dataset": "Fashion-MNIST",
    "model": "FashionCNN",
    "batch_size": 64,
    "learning_rate": 0.001,
    "optimizer": "Adam",
    "epochs": 5,
    "dropout": 0.2
}



def data_loading():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )


    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False
    )

    return train_loader, val_loader

#----------------------------------------------#
#      Model Definition
#----------------------------------------------#
class FashionCNN(nn.Module):

    def __init__(self, dropout=0.3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


#----------------------------------------------#
#      Train one epoch
#----------------------------------------------#
def train_one_epoch(device, model, optimizer, criterion, loader):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(loader)
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


#----------------------------------------------#
#      Validation
#----------------------------------------------#
def validate(device, model, criterion, loader):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / len(loader)
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


#----------------------------------------------#
#      Train
#----------------------------------------------#
def train(train_loader, valid_loader, device, model):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), 
                           lr=config["learning_rate"])

    with mlflow.start_run(run_name="baseline_cnn"):

        mlflow.log_params(config)

        best_val_acc = 0.0

        for epoch in range(config["epochs"]):

            train_loss, train_acc = train_one_epoch(device, model, 
                                optimizer, criterion, train_loader)

            val_loss, val_acc = validate(device, model, criterion, valid_loader)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

            if val_acc > best_val_acc:
                best_val_acc = val_acc

            print(
                f"Epoch {epoch+1}/{config['epochs']} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )

        mlflow.log_metric("best_val_accuracy", best_val_acc)
        mlflow.pytorch.log_model(model, artifact_path="model")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mlflow.set_experiment("fashion_mnist_cnn")

    train_loader, valid_loader = data_loading()

    model = FashionCNN(dropout = config["dropout"]).to(device)

    train(train_loader, valid_loader, device, model)
    

if __name__ == "__main__":
    main()
