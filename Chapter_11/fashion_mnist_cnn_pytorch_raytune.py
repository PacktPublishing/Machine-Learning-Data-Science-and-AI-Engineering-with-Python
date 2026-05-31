#----------------------------------------------#
#       Import packages
#----------------------------------------------#
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from ray import tune



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
#      Ray Tune training function
#----------------------------------------------#
def train_ray_tune(config):
    set_seed(BASE_CONFIG["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = data_loading(
        batch_size=config["batch_size"]
    )

    model = FashionCNN(
        dropout=config["dropout"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = create_optimizer(
        model,
        config["optimizer"],
        config["learning_rate"]
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
            f"Epoch {epoch + 1}/{BASE_CONFIG['epochs']} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        tune.report({
            "val_accuracy": val_acc,
            "val_loss": val_loss,
            "train_accuracy": train_acc,
            "train_loss": train_loss,
            "best_val_accuracy": best_val_acc
        })


#----------------------------------------------#
#      Main
#----------------------------------------------#
def main():

    search_space = {
        "batch_size": tune.choice([32, 64, 128]),

        "learning_rate": tune.loguniform(
            1e-5,
            1e-2
        ),

        "dropout": tune.uniform(
            0.2,
            0.5
        ),

        "optimizer": tune.choice(
            ["Adam", "SGD"]
        )
    }

    tuner = tune.Tuner(
        train_ray_tune,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="best_val_accuracy",
            mode="max",
            num_samples=10
        )
    )

    results = tuner.fit()

    best_result = results.get_best_result(
        metric="best_val_accuracy",
        mode="max"
    )

    print("\nBest validation accuracy:")
    print(best_result.metrics["best_val_accuracy"])

    print("\nBest hyperparameters:")
    print(best_result.config)


if __name__ == "__main__":
    main()