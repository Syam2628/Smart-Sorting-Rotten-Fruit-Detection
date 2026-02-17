import sys
import os
print(f"Python: {sys.version}")
print(f"CWD: {os.getcwd()}")
try:
    import numpy as np
    print("NumPy imported")
    import tensorflow as tf
    print(f"TensorFlow imported: {tf.__version__}")
except ImportError as e:
    print(f"ImportError: {e}")

from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

from model_builder import build_model

app = Flask(__name__)

# Load Model
MODEL_PATH = "healthy_vs_rotten (1).h5"
if not os.path.exists(MODEL_PATH):
    if os.path.exists("healthy_vs_rotten.h5"):
        MODEL_PATH = "healthy_vs_rotten.h5"

print(f"Loading weights from {MODEL_PATH}")
try:
    # Build model manually to avoid shape/deserialization issues
    model = build_model()
    model.load_weights(MODEL_PATH)
    print("Model loaded successfully with manual build!")
    
    # Check output shape
    output_shape = model.output_shape
    print(f"Model output shape: {output_shape}")
except Exception as e:
    print(f"Error loading model: {e}")
    import traceback
    traceback.print_exc()
    model = None

# Class Names
class_names = [
    "Apple_healthy", "Apple_rotten",
    "Banana_healthy", "Banana_rotten",
    "Bellpepper_healthy", "Bellpepper_rotten",
    "Carrot_healthy", "Carrot_rotten",
    "Cucumber_healthy", "Cucumber_rotten",
    "Grape_healthy", "Grape_rotten",
    "Guava_healthy", "Guava_rotten",
    "Mango_healthy", "Mango_rotten",
    "Orange_healthy", "Orange_rotten",
    "Potato_healthy", "Potato_rotten",
    "Strawberry_healthy", "Strawberry_rotten",
    "Tomato_healthy", "Tomato_rotten",
    "Watermelon_healthy", "Watermelon_rotten",
    "Papaya_healthy", "Papaya_rotten"
]

if model and model.output_shape[-1] != len(class_names):
    print(f"WARNING: Model output units ({model.output_shape[-1]}) does not match Class Names count ({len(class_names)})!")
    print("Predictions for classes > 7 will be impossible.")

IMG_SIZE = 100

def predict_image(image):
    try:
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        img = image.resize((IMG_SIZE, IMG_SIZE))
        img_array = img_to_array(img)
        print(f"Image shape before expansion: {img_array.shape}")
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        print(f"Image shape after expansion: {img_array.shape}")

        if model is None:
            print("Error: Model is None")
            return "Model not loaded", 0.0

        print("Starting prediction...")
        prediction = model.predict(img_array)
        print(f"Raw prediction output: {prediction}")
        
        class_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        
        # Safety check for index
        if class_index >= len(class_names):
             return f"Unknown Class {class_index}", confidence
        
        return class_names[class_index], confidence
    except Exception as e:
        print(f"Prediction error details: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", 0.0

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file part")
        
        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No selected file")
        
        if file:
            try:
                img = Image.open(file.stream)
                predicted_class, confidence = predict_image(img)
                return render_template("index.html", prediction=predicted_class, confidence=f"{confidence:.2f}%")
            except Exception as e:
                 return render_template("index.html", error=f"Error processing image: {e}")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
