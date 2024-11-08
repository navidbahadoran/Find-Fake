from flask import Flask, request, render_template
import joblib
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

# Load your trained model (you can save it as a .pkl file after training)
model = joblib.load("stacking_clf_lgbm_meta.pkl")

def preprocess_image(image):
    # Convert image to array, resize, and preprocess as done for your training
    img = image.convert("L")  # Convert to grayscale if needed
    img = img.resize((224, 224))  # Example size; adjust to fit your model's input
    img_array = np.array(img).flatten()  # Flatten if your model expects a 1D array
    return img_array

@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            img = Image.open(io.BytesIO(file.read()))
            img_array = preprocess_image(img)
            prediction = model.predict([img_array])[0]
            result = "Tampered" if prediction == 1 else "Authentic"
            return render_template("result.html", result=result)
    return render_template("upload.html")

if __name__ == "__main__":
  port = int(os.environ.get("PORT", 5000))  # Default to 5000 if PORT not set
  app.run(host="0.0.0.0", port=port)
