# Skin Cancer Detection System

Repository: https://github.com/VirajBarapatre/Skin-Cancer-Detection-System
License: MIT

## Overview

A deep learning-based skin cancer detection system using Convolutional Neural
Networks (CNN) to classify skin cancer images into seven categories, with a
Tkinter GUI for real-time predictions.

Built with TensorFlow and Keras, trained on the HAM10000 dataset.

## Key Features

- Image preprocessing using OpenCV and NumPy
- CNN model for image classification
- Model training and evaluation scripts
- GUI interface (Tkinter) for user interaction — upload an image, get a prediction
- Visualization of training accuracy and loss

## Dataset

Dataset: HAM10000

Classes:
- Melanoma
- Melanocytic Nevi
- Basal Cell Carcinoma
- Actinic Keratoses
- Benign Keratosis
- Dermatofibroma
- Vascular Lesions

## Installation

```
git clone https://github.com/VirajBarapatre/Skin-Cancer-Detection-System.git
cd Skin-Cancer-Detection-System
pip install -r requirements.txt
```

## How to Run

```
python utils/preprocess.py   # Preprocess the dataset
python train_model.py        # Train the model
python evaluate.py           # Evaluate the model
python gui.py                # Run the GUI
```

## Results

Model Accuracy: 90%

## Tech Stack

Python, TensorFlow, Keras, NumPy, OpenCV, Tkinter, Matplotlib

## Author

Developed by Viraj Barapatre. GitHub: https://github.com/VirajBarapatre
