# Image Classification Project

This repository contains a project for image classification using TensorFlow and Keras. The project includes scripts for training a model, deploying an application, and testing the model with sample images.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Testing](#testing)
4. [Files Description](#files-description)


## Installation

### Install Required Libraries

Install the necessary Python libraries listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Usage

Run the Image Classification Script

To train the image classification model, run the Image_Classification.py script:
```bash
python Image_Classification.py
```
This script will:

Load and preprocess images from the specified directory.

Train a convolutional neural network (CNN) on the images.

Evaluate the model's performance and display training and validation loss/accuracy graphs.

Save the trained model to the models directory.

### Deploy the Application

To deploy the application using Streamlit, run the deploy_application.py script:

```bash
streamlit run deploy_application.py
```

### Testing

Change Test Images

You can test the trained model with different images by replacing test_cat.jpg or test_dog.jpg in the Image_Classification.py script with your own images. 
Ensure that the images are placed in the appropriate directory

### Files Description

images/: Directory containing image data for training and testing.

models/: Directory where the trained model is saved (imageclassifier.h5).

Accuracy.png: Plot showing model accuracy over training epochs.

Image_Classification.py: Python script for training the image classification model.

Loss.png: Plot showing model loss over training epochs.

crawl_images.py: Python script for crawling and processing images.

deploy_application.py: Python script for deploying the model with Streamlit.

requirements.txt: File listing required Python libraries.

run_model.png: Plot showing example results from the model.

test_cat.jpg: Sample image of a cat for testing.

test_dog.jpg: Sample image of a dog for testing.

This script will start a Streamlit application that allows you to interact with the trained model.

# Image Classification Web App

## Overview

This is a web application for image classification. You can upload an image of a cat or dog, and the application will display the name of the animal in the image.

## Accessing the Web Application

1. Open your web browser and navigate to the following URL: [Image Classification Web App](https://images-classification.streamlit.app).

## How to Use

1. **Upload an Image:**
   - On the homepage, you will find an option to upload an image.
   - Click on the "Upload" button or drag and drop your image into the designated area.

2. **View Classification Result:**
   - After uploading the image, the application will process it and display the name of the animal found in the image on the screen.


## Troubleshooting

- If you encounter issues with uploading or image processing, ensure that the image format is supported and try refreshing the page.

## Note

Please be aware that the model used for classification is trained with a limited number of images. As a result, the accuracy of the classification might not always be perfect. The model may occasionally produce incorrect or unexpected results due to the constraints in its training data. If you find that the classification is not as accurate as expected, it may be due to the model's limited exposure to diverse examples during training. We appreciate your understanding and encourage you to provide feedback if you encounter any issues with the classification results.



