
import os
from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load your saved model
model = load_model('skin_new_model.h5',compile=False)

# Define class labels based on your dataset
CLASS_LABELS = {
    0: "Actinic Keratoses",
    1: "Basal Cell Carcinoma",
    2: "Benign Keratosis-like Lesions",
    3: "Dermatofibroma",
    4: "Melanoma",
    5: "Malanoma",
    6: "Vascular Lesions"
} # Replace with actual class names

def preprocess_image(image_path):
    # Load the image and preprocess
    img = load_img(image_path, target_size=(150, 150))  # Match model's input size
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize to match training data
    return img

@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        # Save the uploaded file
        file = request.files["file"]
        if file:
            file_path = os.path.join("static", file.filename)
            file.save(file_path)

            # Preprocess and predict
            img = preprocess_image(file_path)
            predictions = model.predict(img)
            predicted_class = CLASS_LABELS[np.argmax(predictions[0])]

            return render_template("result.html", prediction=predicted_class, image=file_path)
    return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)

