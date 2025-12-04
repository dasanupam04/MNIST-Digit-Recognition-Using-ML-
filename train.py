# src/train.py
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from .pytorch_cnn import SmallCNN
from .utils import set_seed


def get_dataloaders(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test  = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return running_loss / total, correct / total

def validate(model, loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss = criterion(out, y)

            running_loss += loss.item() * X.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return running_loss / total, correct / total

def main(args):
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_dataloaders(args.batch_size)

    model = SmallCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    os.makedirs(os.path.dirname(args.checkpoint_path) or ".", exist_ok=True)

    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc     = validate(model, test_loader, criterion, device)

        print(f"Epoch {epoch}/{args.epochs}  "
              f"Train Acc: {train_acc:.4f}  Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.checkpoint_path)
            print(f"Checkpoint saved at: {args.checkpoint_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/model.pt")
    args = parser.parse_args()
    main(args)
