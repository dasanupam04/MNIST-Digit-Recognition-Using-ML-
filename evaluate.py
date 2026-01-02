import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from .pytorch_cnn import SmallCNN
from .utils import set_seed

def load_data(batch_size=256):
    transform = transforms.Compose([transforms.ToTensor()])
    test = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    return test, test_loader

def evaluate_model(checkpoint_path="checkpoints/model.pt", outdir="evaluation_output"):
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = SmallCNN().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Data
    test_dataset, test_loader = load_data()

    all_preds = []
    all_labels = []

    # Prediction loop
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            preds = out.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Accuracy
    acc = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {acc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

    # Save misclassified images
    import os
    os.makedirs(outdir, exist_ok=True)

    mis_idx = np.where(all_preds != all_labels)[0][:10]  # first 10 misclassified
    for i, idx in enumerate(mis_idx):
        img = test_dataset[idx][0].squeeze().numpy()
        true_label = all_labels[idx]
        pred_label = all_preds[idx]
        plt.imsave(f"{outdir}/mis_{i}_true{true_label}_pred{pred_label}.png", img, cmap="gray")

    print(f"Saved {len(mis_idx)} misclassified images to {outdir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="checkpoints/model.pt")
    parser.add_argument("--outdir", default="evaluation_output")
    args = parser.parse_args()

    evaluate_model(checkpoint_path=args.model, outdir=args.outdir)

