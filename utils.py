import cv2
import numpy as np
from PIL import Image
import os

def save_temp_image(uploaded_file):
    temp_path = "temp.jpg"
    uploaded_file.save(temp_path)
    return temp_path

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(image, 100, 200)
    return Image.fromarray(edges)
