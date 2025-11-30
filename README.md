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



Model saves to:

    checkpoints/model.pt



🔍 Inference API

Start the backend:


    python -m api.app


Endpoints

✔ Health check


    GET /ping


POST /predict

✔ Predict

    POST /predict




Accepts:

file (multipart/form-data)

OR

JSON base64 image

Returns:

     {
     "prediction": 7,
     "probs": [0.01, 0.02, ...]
     }



✔ Save a Correction

    POST /corrections




Collects:

    1. correct/incorrect
    2. true label
    3. original prediction
    4. uploaded/drawn image




🎨 Frontend (User Interface)

Serve the UI locally:

    python -m http.server 8000



Open:

    http://127.0.0.1:8000/frontend.html




Frontend features:

    1. Preview uploaded images
    2. Draw using mouse/touch
    3. Visual prediction bars
    4. Confidence score
    5. Correction buttons:
    . This is correct
    . This is incorrect (enter true label)

    

Canvas Preprocessing


The drawing is:

     1. Cropped to bounding box
     2. Scaled to 20×20
     3. Centered in 28×28
     4. Lightly blurred
     5. Sent to the API

This dramatically improves model accuracy on hand-drawn digits.




