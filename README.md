# MNIST-Digit-Recognition-Using-ML-
This project transforms the classic MNIST dataset into a full production-style application that can recognize hand-drawn digits with high accuracy — whether uploaded as an image or drawn directly on a canvas. It goes far beyond a simple notebook: every part of the ML workflow is implemented, automated, and connected.

It includes:

   1. ✔ A PyTorch CNN for digit classification
   2. ✔ Evaluation & misclassification analysis
   3. ✔ A Flask inference API
   4. ✔ A fully interactive frontend (upload + drawing canvas)
   5. ✔ Preprocessing pipeline for hand-drawn digits
   6. ✔ Human-in-the-loop feedback (Correct / Incorrect)
   7. ✔ Data collection for future retraining


🚀 Features
🧠 Trained PyTorch CNN Model

A lightweight convolutional neural network trained on MNIST, achieving high accuracy on test data.

🎨 Interactive Frontend (HTML + JS)
    1. Upload any digit image
    2. Or draw directly using a canvas
    3. Automatic preprocessing (crop → scale → pad → smooth)
    4. Displays prediction + probability bars
