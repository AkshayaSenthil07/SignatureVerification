# SignatureVerification
Signature Verification System is a machine learning-based project that automates the process of authenticating handwritten signatures. Built using Python, Scikit-learn, and Gradio, this system processes scanned signature images, extracts distinctive features, and classifies them as genuine or forged with accuracy percentage

Overview:
This project presents a Signature Verification System designed to identify and verify handwritten signatures using machine learning algorithms. The system addresses fraud prevention and identity verification challenges by distinguishing between genuine and forged signatures.

It utilizes Scikit-learn for model training, Scikit-image and Pillow for preprocessing and feature extraction, and Gradio to build an interactive web interface where users can upload signature images and receive instant verification results.

Features:
1. Processes and classifies signature images as authorised or unauthorised

2. Performs grayscale conversion, resizing, and pixel feature extraction for training

3. Uses a Logistic Regression model for classification

4. Provides real-time predictions with confidence scores

5. Includes a Gradio web app for user interaction and result display

Technical Stack:

1. Programming Language: Python

2. Libraries Used: Scikit-learn, Scikit-image, Pillow, Torch, Gradio, NumPy

3. Model Type: Logistic Regression (binary classification)

4. Dataset: Custom-organized authentic and forged signature samples

Results: 
The model demonstrates high classification accuracy and a clean output interface that immediately returns “Authorised” or “Unauthorised” depending on the input image.

Possible Extensions:
1. Integration with Deep Learning (CNN) models for higher precision.

2. Linking digital verification with organisational workflows for real-time authentication systems.

