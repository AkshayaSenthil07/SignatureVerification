from google.colab import drive
drive.mount('/content/drive')
model_path = "/content/drive/MyDrive/path_to_model/signature_py_model"

# ======================================================
# ✅ STEP 1 — Install Required Libraries
# ======================================================
!pip install torch torchvision scikit-learn gradio pillow --quiet

# ======================================================
# ✅ STEP 2 — Import Libraries
# ======================================================
import torch
from torchvision import transforms
from PIL import Image
import gradio as gr
import sklearn.linear_model  # Needed for safe unpickling
import numpy as np # Import numpy

# ======================================================
# ✅ STEP 3 — Mount Google Drive
# ======================================================
from google.colab import drive
drive.mount('/content/drive')

# ======================================================
# ✅ STEP 4 — Load Your Saved LogisticRegression Model
# ======================================================
# Change this path to match your Drive location
model_path = "/content/drive/MyDrive/signature_py_model"  # Add .pkl if your file has one

# Allow safe unpickling for LogisticRegression and numpy.ndarray
torch.serialization.add_safe_globals([sklearn.linear_model._logistic.LogisticRegression])
torch.serialization.add_safe_globals([np.ndarray])


# Load your model, allowing non-weights to be loaded
model = torch.load(model_path, map_location="cpu", weights_only=False)
print("✅ Scikit-Learn model loaded successfully!")
 ======================================================
# ✅ STEP 5 — Define Image Preprocessing (Adjusted for 100x100 grayscale)
# ======================================================
transform = transforms.Compose([
    transforms.Resize((100, 100)),  # 👈 must match your training size
    transforms.Grayscale(),
    transforms.ToTensor()
])


# ======================================================
# ✅ STEP 6 — Define Prediction Function
# ======================================================
def predict_signature(image):
    try:
        img = image.convert("RGB")
        img_t = transform(img).unsqueeze(0)  # Shape: [1, C, H, W]

        # Flatten for sklearn model (since it expects 1D feature vectors)
        img_flat = img_t.view(1, -1).numpy()

        # Run the sklearn model
        # Check if the model has predict_proba
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(img_flat)[0]
            pred = model.predict(img_flat)[0]
            confidence = max(prob)
        elif hasattr(model, 'predict'):
             # If no predict_proba, just use predict and set confidence to 1.0
             pred = model.predict(img_flat)[0]
             confidence = 1.0 # Or some other default if needed
        else:
            return "⚠️ Error: Loaded model does not have 'predict' or 'predict_proba' method."


        # Labels (adjust if swapped)
        labels = ["unauthorised", "authorised"]

        # Ensure pred is an integer index
        predicted_label = labels[int(pred)]


        return {predicted_label: float(confidence)}

    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# ======================================================
# ✅ STEP 7 — Create Gradio Interface
# ======================================================
interface = gr.Interface(
    fn=predict_signature,
    inputs=gr.Image(type="pil", label="Upload a Signature Image"),
    outputs=gr.Label(num_top_classes=2, label="Prediction"),
    title="Signature Verification Web App",
    description="Upload a signature image to check if it's authorised or unauthorised."
)

# ======================================================
# ✅ STEP 8 — Launch App
# ======================================================
interface.launch(share=True)
