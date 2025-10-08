import io
import os
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

# Initialize the FastAPI app
app = FastAPI()

# --- Enable CORS (Cross-Origin Resource Sharing) ---
# This is the most important change for Render.
# Because your frontend and backend will be on different URLs, this code block
# acts as a security pass, allowing your frontend to make API requests to this backend.
# Without this, your browser would block the requests for security reasons.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins. For production, you can restrict this to your frontend's URL.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model and Application Constants ---
# This uses pathlib to create a robust path to the model file.
MODEL_PATH = Path(__file__).parent / "cifar10_mobilenetv2_model.h5"
IMG_SIZE = 96
CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# --- Load the Trained Model ---
model = None
if MODEL_PATH.exists():
    try:
        model = tf.keras.models.load_model(str(MODEL_PATH))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"FATAL: Model could not be loaded. Error: {e}")
else:
    print(f"FATAL: Model file not found at {MODEL_PATH}")

# --- Preprocessing Function (Corrected) ---
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    This function takes raw image bytes, decodes them, resizes, and most importantly,
    applies the specific preprocessing required by the MobileNetV2 model.
    """
    image = tf.image.decode_image(image_bytes, channels=3)
    image_resized = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    img_array_expanded = tf.expand_dims(image_resized, 0)
    # This final step formats the image data (e.g., scales pixel values)
    # exactly how the model was trained, which is crucial for accuracy.
    return tf.keras.applications.mobilenet_v2.preprocess_input(img_array_expanded)

# --- API Endpoints ---
@app.get("/")
def read_root():
    """A simple endpoint to visit in your browser to confirm the API is running."""
    return {"message": "CIFAR-10 Classifier API is running."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Receives an image file, preprocesses it, and returns the model's prediction."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded on the server.")
    
    # Read the file content
    image_bytes = await file.read()
    
    # Process the image
    processed_image = preprocess_image(image_bytes)
    
    # Make a prediction
    predictions = model.predict(processed_image)
    
    # Extract the top prediction and its confidence
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    confidence = float(np.max(predictions[0]))

    return {
        "predicted_class": predicted_class_name,
        "confidence": f"{confidence:.2%}" # Format as a percentage string
    }

