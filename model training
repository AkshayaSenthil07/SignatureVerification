from google.colab import files
import zipfile
import os

print("Upload your dataset zip file (e.g., signatures.zip):")
uploaded = files.upload()  # Select your ZIP file

zip_path = list(uploaded.keys())[0]  # Get uploaded filename
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("/content/dataset")

print("Folders inside '/content/dataset/data':", os.listdir("/content/dataset/data"))

genuine_path = "/content/dataset/data/genuine/full_org"
forged_path  = "/content/dataset/data/forged/full_forg"

print("Number of genuine samples:", len(os.listdir(genuine_path)))
print("Number of forged samples:", len(os.listdir(forged_path)))

import glob
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
import numpy as np
import os


def extract_features(img_path):
    img = imread(img_path)
    # If grayscale
    if img.ndim == 2:
        img_gray = img
    # If RGB
    elif img.ndim == 3:
        if img.shape[2] == 3:      # Standard RGB
            img_gray = rgb2gray(img)
        elif img.shape[2] == 4:    # RGBA
            img_gray = rgb2gray(img[..., :3])  # Drop alpha channel, use RGB only
        else:
            raise ValueError(f"Unsupported channel shape: {img.shape}")
    else:
        raise ValueError(f"Unsupported image shape {img.shape} for {img_path}")
    img_resized = resize(img_gray, (100, 100))
    return img_resized.flatten()


def load_images_from_folder(folder):
    features = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        for file in glob.glob(os.path.join(folder, "**", ext), recursive=True):
            try:
                features.append(extract_features(file))
            except Exception as e:
                print("Error loading:", file, e)
    return features

genuine_features = load_images_from_folder(genuine_path)
forged_features = load_images_from_folder(forged_path)

print("Extracted genuine features:", len(genuine_features))
print("Extracted forged features:", len(forged_features))

X = np.vstack(genuine_features + forged_features)
y = np.array([1]*len(genuine_features) + [0]*len(forged_features))

print("Total samples:", X.shape[0])
print("Feature dimension:", X.shape[1])
print("Labels shape:", y.shape)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Train samples:", X_train.shape[0])
print("Test samples:", X_test.shape[0])

import os
import glob
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
import numpy as np

# Assuming genuine_path is defined and contains genuine images
# Use the first genuine image found as an example
genuine_images = glob.glob(os.path.join(genuine_path, "**", "*.png"), recursive=True) + \
                 glob.glob(os.path.join(genuine_path, "**", "*.jpg"), recursive=True) + \
                 glob.glob(os.path.join(genuine_path, "**", "*.jpeg"), recursive=True)


if genuine_images:
    new_img_path = genuine_images[0]
    print(f"Using genuine image for prediction: {new_img_path}")
else:
    print("No genuine images found in the specified path.")
    new_img_path = None # Set to None if no image is found

def extract_features(img_path):
    img = imread(img_path)
    if img.ndim == 2:
        img_gray = img
    elif img.ndim == 3:
        if img.shape[2] == 3:
            img_gray = rgb2gray(img)
        elif img.shape[2] == 4:
            img_gray = rgb2gray(img[..., :3])
        else:
            raise ValueError(f"Unsupported channel shape: {img.shape}")
    else:
        raise ValueError(f"Unsupported image shape {img.shape} for {img_path}")
    img_resized = resize(img_gray, (100, 100))
    return img_resized.flatten()

# Load and preprocess the image, then predict
if new_img_path:
    new_features = extract_features(new_img_path).reshape(1, -1)
    prediction = model.predict(new_features)[0]

    if prediction == 1:
        print("Prediction: Genuine signature")
    else:
        print("Prediction: Forged signature")

uploaded_files = files.upload()

# Step 2: Get the uploaded file path
for filename in uploaded_files.keys():
    uploaded_img_path = filename  # file is now saved in /content/ folder in Colab

# Step 3: Extract features from the uploaded image (same preprocessing)
def extract_features(img_path):
    img = imread(img_path)
    if img.ndim == 2:
        img_gray = img
    elif img.ndim == 3:
        if img.shape[2] == 3:
            img_gray = rgb2gray(img)
        elif img.shape[2] == 4:
            img_gray = rgb2gray(img[..., :3])
    else:
        raise ValueError(f"Unexpected image shape: {img.shape}")
    img_resized = resize(img_gray, (100, 100))
    return img_resized.flatten()

# Step 4: Predict using the trained model
features = extract_features(uploaded_img_path).reshape(1, -1)
pred = model.predict(features)

# Step 5: Display results
if pred == 1:
    print(f"The signature in '{uploaded_img_path}' is predicted as: Genuine")
else:
    printf("The signature is '{uploaded_img_path}' is predicted as: forged")
