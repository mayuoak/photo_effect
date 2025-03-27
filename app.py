import os
import torch
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from flask_ngrok import run_with_ngrok
from models import GhibliModel
from utils import save_temp_image, preprocess_image
from io import BytesIO

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)
run_with_ngrok(app)

# Load Ghibli model
ghibli_model = GhibliModel()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image_path = save_temp_image(file)

    # Process and generate output
    processed_image = preprocess_image(image_path)
    output = ghibli_model.generate_image(processed_image)

    # Convert to BytesIO for sending response
    img_io = BytesIO()
    output.save(img_io, "JPEG")
    img_io.seek(0)

    return send_file(img_io, mimetype="image/jpeg")

if __name__ == "__main__":
    app.run()
