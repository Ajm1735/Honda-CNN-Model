Honda CNN Model — Image Classification on 5 Honda Car Models

This project is a computer vision model built using a Convolutional Neural Network (CNN) to classify images of five Honda car models:

Civic

Odyssey

CR-V

Element

Ridgeline

The model is trained on a manually collected dataset of images downloaded from the web.
It uses TensorFlow/Keras and is designed to run smoothly in Google Colab.

📁 Dataset Download

The dataset is too large for GitHub’s normal upload limits, so it is hosted as a GitHub Release asset.

📥 Download the dataset ZIP here:
👉 https://github.com/Ajm1735/Honda-CNN-Model/releases/latest

Download this file:

OD.CIV.HRV.RIDGELINE.ELEMENT.FINAL.zip


This ZIP contains the full folder structure:

train/
val/
test/

🚀 How to Run the Model in Google Colab

Follow these steps to run the project in Google Colab.

✅ 1. Open the Notebook in Google Colab

Go to the repository
https://github.com/Ajm1735/Honda-CNN-Model

Click the notebook file:
Final_449_Honda.ipynb

Click “Open in Colab”

✅ 2. Download the Dataset

Download the ZIP from the Release page:

📥 https://github.com/Ajm1735/Honda-CNN-Model/releases/latest

File name:

OD.CIV.HRV.RIDGELINE.ELEMENT.FINAL.zip

✅ 3. Upload the Dataset to Colab

Run the notebook cells until you reach:

from google.colab import files
uploaded = files.upload()


When prompted, upload:

OD.CIV.HRV.RIDGELINE.ELEMENT.FINAL.zip

✅ 4. Extract the Dataset

The notebook automatically unzips the dataset and creates:

/content/data/
    train/
    val/
    test/


No manual setup is required.

✅ 5. Train the CNN

Run the remaining cells in order:

Preprocessing

Data augmentation

Model creation

Training

Evaluation

The model will output accuracy, loss, and prediction examples.

🧠 Model Overview

The model uses:

TensorFlow/Keras

ImageDataGenerator for preprocessing

Convolutional + MaxPooling layers

Dense layers for classification

Softmax output for 5-class prediction

Training includes:

Data augmentation

Early stopping

Learning rate adjustments (if implemented)

📊 Results (Fill These In After Running)

Example:

Training Accuracy: XX%

Validation Accuracy: XX%

Test Accuracy: XX%

Best Performing Class: (Ex: Element, Ridgeline)

Confusion Matrix: (if applicable)

📦 Repository Structure
Honda-CNN-Model/
│
├── Final_449_Honda.ipynb     <-- Main notebook
├── data/                      <-- Created automatically in Colab after unzipping
│   ├── train/
│   ├── val/
│   └── test/
└── README.md

🙌 Credits

Dataset collected manually from publicly available web images.
Model created by Aiden Michael (Ajm1735) for ISAT 449.
