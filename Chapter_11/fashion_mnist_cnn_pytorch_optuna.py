#----------------------------------------------#
#       Import packages
#----------------------------------------------#
import torch
import torch.nn as nn
import torch.optim as optim
import optuna

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


#----------------------------------------------#
#      Base configuration
#----------------------------------------------#
BASE_CONFIG = {
    "dataset": "Fashion-MNIST",
    "model": "FashionCNN",
    "epochs": 5,
    "seed": 42,
    "train_split": 0.8
}


#----------------------------------------------#
#      Reproducibility
#----------------------------------------------#
def set_seed(seed=42):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


#----------------------------------------------#
#      Data loading
#----------------------------------------------#
def data_loading(batch_size):
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

    train_size = int(BASE_CONFIG["train_split"] * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(BASE_CONFIG["seed"])

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
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
#      Optimizer factory
#----------------------------------------------#
def create_optimizer(model, optimizer_name, learning_rate):

    if optimizer_name == "Adam":
        return optim.Adam(
            model.parameters(),
            lr=learning_rate
        )

    if optimizer_name == "SGD":
        return optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9
        )

    raise ValueError(f"Unknown optimizer: {optimizer_name}")


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
#      Optuna objective function
#----------------------------------------------#
def objective(trial):
    set_seed(BASE_CONFIG["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = trial.suggest_categorical(
        "batch_size",
        [32, 64, 128]
    )

    learning_rate = trial.suggest_float(
        "learning_rate",
        1e-5,
        1e-2,
        log=True
    )

    dropout = trial.suggest_float(
        "dropout",
        0.2,
        0.5
    )

    optimizer_name = trial.suggest_categorical(
        "optimizer",
        ["Adam", "SGD"]
    )

    train_loader, val_loader = data_loading(
        batch_size=batch_size
    )

    model = FashionCNN(
        dropout=dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = create_optimizer(
        model,
        optimizer_name,
        learning_rate
    )

    best_val_acc = 0.0

    for epoch in range(BASE_CONFIG["epochs"]):

        train_loss, train_acc = train_one_epoch(
            device,
            model,
            optimizer,
            criterion,
            train_loader
        )

        val_loss, val_acc = validate(
            device,
            model,
            criterion,
            val_loader
        )

        best_val_acc = max(best_val_acc, val_acc)

        print(
            f"Trial {trial.number} | "
            f"Epoch {epoch + 1}/{BASE_CONFIG['epochs']} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

    return best_val_acc


#----------------------------------------------#
#      Main
#----------------------------------------------#
def main():
    study = optuna.create_study(
        study_name="fashion_mnist_cnn",
        direction="maximize",
        storage="sqlite:///optuna_fashion_mnist.db",
        load_if_exists=True
    )

    study.optimize(
        objective,
        n_trials=10
    )

    print("\nBest trial:")
    print(study.best_trial.number)

    print("\nBest validation accuracy:")
    print(f"{study.best_value:.4f}")

    print("\nBest hyperparameters:")
    print(study.best_params)


if __name__ == "__main__":
    main()