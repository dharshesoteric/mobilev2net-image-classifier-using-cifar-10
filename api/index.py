import io
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# Initialize the FastAPI app
app = FastAPI(title="CIFAR-10 Image Classifier API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model and Application Constants ---
MODEL_PATH = 'cifar10_mobilenetv2_model.h5'
IMG_SIZE = 96
CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# --- Load the Trained Model ---
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    model = None
    print(f"FATAL: Model could not be loaded. Error: {e}")

# --- Preprocessing Function ---
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet_v2.preprocess_input(img_array_expanded)

# --- API Endpoints ---

# === THIS IS THE KEY CHANGE ===
# This endpoint now reads and returns your index.html file.
@app.get("/", response_class=HTMLResponse)
async def read_index():
    """Serves the frontend HTML file."""
    try:
        with open("index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="index.html not found")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Receives an image and returns a prediction."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
    
    try:
        image_bytes = await file.read()
        processed_image = preprocess_image(image_bytes)
        predictions = model.predict(processed_image)
        
        predicted_class_index = np.argmax(predictions[0])
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        confidence = float(np.max(predictions[0]))

        return {
            "predicted_class": predicted_class_name,
            "confidence": f"{confidence:.2%}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

