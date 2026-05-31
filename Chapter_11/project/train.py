import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import mlflow
import mlflow.pytorch

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
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_data_loaders(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    dataset = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )

    train_size = int(config["train_split"] * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(config["seed"])

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
    )

    return train_loader, val_loader


def create_optimizer(model, config):
    if config["optimizer"] == "Adam":
        return optim.Adam(
            model.parameters(),
            lr=config["learning_rate"],
        )

    if config["optimizer"] == "SGD":
        return optim.SGD(
            model.parameters(),
            lr=config["learning_rate"],
            momentum=0.9,
        )

    raise ValueError(f"Unknown optimizer: {config['optimizer']}")


def train_one_epoch(device, model, optimizer, criterion, loader):
    model.train()

    total_loss = 0.0
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

        total_loss += loss.item()

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


def validate(device, model, criterion, loader):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total


def run_training(config):
    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_data_loaders(config)

    model = FashionCNN(
        dropout=config["dropout"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, config)

    mlflow.set_experiment("fashion_mnist_cnn")

    best_val_acc = 0.0

    with mlflow.start_run(run_name=config.get("run_name", "fashion_cnn_run")):

        mlflow.log_params(config)

        for epoch in range(config["epochs"]):
            train_loss, train_acc = train_one_epoch(
                device,
                model,
                optimizer,
                criterion,
                train_loader,
            )

            val_loss, val_acc = validate(
                device,
                model,
                criterion,
                val_loader,
            )

            best_val_acc = max(best_val_acc, val_acc)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

            print(
                f"Epoch {epoch + 1}/{config['epochs']} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )

        mlflow.log_metric("best_val_accuracy", best_val_acc)

        model_path = Path(config["model_path"])
        model_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": config,
                "best_val_accuracy": best_val_acc,
            },
            model_path,
        )

        mlflow.log_artifact(str(model_path))
        mlflow.pytorch.log_model(model, artifact_path="model")

    print(f"\nSaved model to: {model_path}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    return best_val_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    run_training(config)


if __name__ == "__main__":
    main()