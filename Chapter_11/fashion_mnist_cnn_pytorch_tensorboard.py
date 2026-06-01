#----------------------------------------------#
#       Import packages
#----------------------------------------------#
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import torchvision


#----------------------------------------------#
#      Configuration and Data loading
#----------------------------------------------#

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

    def __init__(self):
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
            nn.Dropout(0.3),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x




#----------------------------------------------#
#      Loss & Optimization
#----------------------------------------------#



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
#      Prepare writer
#----------------------------------------------#
def prepare_writer(train_loader, model, device):
    writer = SummaryWriter(
        log_dir="runs/fashion_baseline"
    )

    images, labels = next(iter(train_loader))

    image_grid = torchvision.utils.make_grid(
        images[:16],
        normalize=True
    )

    # Add sample images
    writer.add_image(
        "FashionMNIST Samples",
        image_grid
    )

    # Add model info
    writer.add_graph(
        model,
        images[:1].to(device)
    )


    return writer


#----------------------------------------------#
#      Train
#----------------------------------------------#
def train(train_loader, valid_loader, device, model, writer):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5

    for epoch in range(num_epochs):

        train_loss, train_acc = train_one_epoch(device, model, 
                            optimizer, criterion, train_loader)

        val_loss, val_acc = validate(device, model, criterion, valid_loader)

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Validation", val_acc, epoch)

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        for name, param in model.named_parameters():
            writer.add_histogram(name, param, epoch)


    writer.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, valid_loader = data_loading()

    model = FashionCNN().to(device)

    writer = prepare_writer(train_loader, model, device)

    train(train_loader, valid_loader, device, model, writer)
    

if __name__ == "__main__":
    main()
