# src/numpy_nn.py
"""
Pure-NumPy educational MLP for MNIST.

Usage:
    python src/numpy_nn.py --epochs 5 --batch-size 256 --lr 0.1 --hidden-size 128 --out numpy_model.npz

Notes:
- Single hidden layer with ReLU, softmax output, cross-entropy loss.
- SGD without momentum (simple and clear).
- Saves weights to .npz for later inspection/evaluation.
"""

import argparse
import numpy as np
import os
from .utils import load_mnist_numpy, set_seed, to_image_numpy

def one_hot(y, num_classes=10):
    y = np.array(y, dtype=np.int64)
    out = np.zeros((y.size, num_classes), dtype=np.float32)
    out[np.arange(y.size), y] = 1.0
    return out

class SimpleFFNN:
    def __init__(self, input_size=784, hidden_size=128, output_size=10, seed=42):
        np.random.seed(seed)
        self.W1 = np.random.randn(input_size, hidden_size).astype(np.float32) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((hidden_size,), dtype=np.float32)
        self.W2 = np.random.randn(hidden_size, output_size).astype(np.float32) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((output_size,), dtype=np.float32)

    def forward(self, x):
        z1 = x.dot(self.W1) + self.b1        # (N, hidden)
        a1 = np.maximum(0, z1)               # ReLU
        z2 = a1.dot(self.W2) + self.b2       # (N, 10)
        z2m = z2 - np.max(z2, axis=1, keepdims=True)
        exp = np.exp(z2m)
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        return z1, a1, z2, probs

    def predict(self, x):
        _, _, _, probs = self.forward(x)
        return np.argmax(probs, axis=1)

    def compute_loss_and_grads(self, x, y_onehot):
        z1, a1, z2, probs = self.forward(x)
        N = x.shape[0]
        loss = -np.sum(y_onehot * np.log(probs + 1e-12)) / N
        dz2 = (probs - y_onehot) / N         # (N,10)
        dW2 = a1.T.dot(dz2)                  # (hidden,10)
        db2 = np.sum(dz2, axis=0)
        da1 = dz2.dot(self.W2.T)             # (N,hidden)
        dz1 = da1 * (z1 > 0)                 # (N,hidden)
        dW1 = x.T.dot(dz1)                   # (784,hidden)
        db1 = np.sum(dz1, axis=0)
        return loss, dW1, db1, dW2, db2

    def step(self, grads, lr):
        dW1, db1, dW2, db2 = grads
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

def train(args):
    set_seed(args.seed)
    (X_train, y_train), (X_test, y_test) = load_mnist_numpy(normalize=True, flatten=True)
    print("Train samples:", X_train.shape[0], "Test samples:", X_test.shape[0])
    model = SimpleFFNN(input_size=784, hidden_size=args.hidden_size, output_size=10, seed=args.seed)
    y_train_one = one_hot(y_train, 10)

    N = X_train.shape[0]
    batch = args.batch_size
    for epoch in range(1, args.epochs + 1):
        perm = np.random.permutation(N)
        X_sh = X_train[perm]
        y_sh = y_train_one[perm]
        epoch_loss = 0.0
        for i in range(0, N, batch):
            xb = X_sh[i:i+batch]
            yb = y_sh[i:i+batch]
            loss, dW1, db1, dW2, db2 = model.compute_loss_and_grads(xb, yb)
            epoch_loss += loss * xb.shape[0]
            model.step((dW1, db1, dW2, db2), args.lr)
        epoch_loss /= N
        preds = model.predict(X_test)
        acc = (preds == y_test).mean()
        print(f"Epoch {epoch}/{args.epochs}  loss={epoch_loss:.4f}  test_acc={acc:.4f}")
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    np.savez(args.out, W1=model.W1, b1=model.b1, W2=model.W2, b2=model.b2)
    print("Saved numpy model to:", args.out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--out', type=str, default='numpy_model.npz')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    train(args)
