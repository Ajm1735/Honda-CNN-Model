Honda Vehicle Classification (MobileNetV2 CNN)

A deep learning project for classifying five Honda car models using a Convolutional Neural Network (CNN) built with TensorFlow/Keras.
This project was created by Aiden Michael (Ajm1735) and Kirolos Boules for ISAT 449.

The model can distinguish between:

Honda Civic

Honda CR-V

Honda Element

Honda Odyssey

Honda Ridgeline

All images were manually collected from the web and organized into training, validation, and test sets.

🚀 Project Versions

This repository provides two complete ways to run the model:

1️⃣ Google Colab Version (Cloud Execution)

Notebook: Final_449_Honda (1).ipynb
Dataset: OD.CIV.HRV.RIDGELINE.ELEMENT.FINAL.zip (Release Asset)

This version requires no local installation — it runs entirely in Google Colab.

How to Use the Colab Version

Download the dataset ZIP from the Releases page:

OD.CIV.HRV.RIDGELINE.ELEMENT.FINAL.zip


Open the notebook in Google Colab.

Early in the notebook, upload the dataset ZIP when prompted.

The notebook automatically:

Extracts the dataset

Organizes the folders (train/, val/, test/)

Prepares the data pipeline

Run the notebook top-to-bottom to:

preprocess data

apply augmentation

train the CNN

evaluate model accuracy

2️⃣ Offline Jupyter Notebook Version (Anaconda)

Notebook: Final_449_Honda.ipynb
Offline Package: Run in Notebook version.zip (Release Asset)

This version is designed for users who prefer local execution using Anaconda.

How to Use the Offline Version

Download:

Run in Notebook version.zip


Unzip the folder anywhere on your computer.

Open Anaconda Navigator → Jupyter Notebook.

Navigate to the unzipped folder.

Open:

Final_449_Honda.ipynb


Run the notebook — all dataset files and model weights are already included.

The offline package contains:

Complete cleaned dataset (data_honda/)

Full training notebook

Phase 1 trained model (honda_mnv2_head.keras)

Fine-tuned final model (honda_mnv2_finetuned.keras)

All scripts needed for offline training and inference

🧠 Model Architecture

This project uses MobileNetV2 (ImageNet-pretrained) as the feature extractor.

Pipeline Overview

Data Augmentation Layer

RandomFlip

RandomRotation

RandomZoom

RandomTranslation

RandomContrast

Rescaling

Normalize pixel values to [0, 1]

MobileNetV2 Backbone

Loaded with include_top=False

Frozen during Phase 1 training

Custom Classification Head

Global Average Pooling

Batch Normalization

Dense (512) + L2 Regularization + Dropout(0.5)

Dense (256) + L2 Regularization + Dropout(0.4)

Softmax output (5 classes)

📊 Training Process
Phase 1 — Train Classification Head

Backbone frozen

Adam optimizer, LR = 1e-4

Callbacks:

ModelCheckpoint

EarlyStopping

ReduceLROnPlateau

Phase 2 — Fine-Tune Full Network

Unfreeze last MobileNetV2 layers

Lower learning rate (1e-5)

Continue training to improve accuracy

📁 Dataset Information

The dataset consists of manually collected images for each vehicle class and is structured as:

data_honda/
    train/
    val/
    test/


All images were validated, cleaned, and converted to TensorFlow-safe formats.

👥 Authors

Aiden Michael (Ajm1735)

Kirolos Boules

Created for ISAT 449 – Deep Learning Project
