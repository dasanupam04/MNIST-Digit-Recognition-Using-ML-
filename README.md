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

  🌐 Flask API (backend)
   1. /predict → model inference
   2. /ping → health check
   3. /corrections → store user feedback (correct/incorrect)
CORS enabled for browser usage.


  📈 Evaluation Tools
   1. Accuracy on MNIST
   2. Confusion matrix
   3. Saves misclassified samples


  👍 Human-in-the-loop Learning
   Users can mark predictions as:
   1. Correct → save as future positive example
   2. Incorrect → user provides correct label
  

📁 Project Structure


    ├── api/
    │   └── app.py               # Flask API (predict + corrections)
    │
    ├── src/
    │   ├── pytorch_cnn.py       # CNN model definition
    │   ├── train.py             # PyTorch training script
    │   ├── utils.py             # Utilities (seed, MNIST loader)
    │   └── numpy_nn.py          # Educational NumPy neural network
    │
    ├── evaluation_output/
    │   ├── confusion_matrix.png
    │   ├── mis_*.png            # Sample misclassified images
    │
    ├── corrections/             # User feedback data gets saved here
    │   └── meta.csv
    │
    ├── checkpoints/
    │   └── model.pt             # Trained PyTorch model
    │
    ├── frontend.html            # Interactive UI (upload + draw)
    └── README.md                # <— You are here


🧪 Model Training

To train (or retrain) the CNN:

   python -m src.train --epochs 5 --batch-size 128 --lr 0.01 --checkpoint-path checkpoints/model.pt

   


   
