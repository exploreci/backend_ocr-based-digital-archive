import pytesseract
import cv2
import numpy as np
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'F:\Program Files\Tesseract-OCR\tesseract.exe'


# ================================================================
# preprocess_for_tesseract
# Konversi gambar ke format optimal sebelum dibaca Tesseract:
#   grayscale → adaptive threshold
# ================================================================
def preprocess_for_tesseract(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.adaptiveThreshold(
        image, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21, 10
    )
    return image


# ================================================================
# extract_text
# Ekstrak teks dari gambar menggunakan Tesseract OCR
# Input : image (numpy array, output dari preprocessing)
# Output: string teks hasil OCR
# ================================================================
def extract_text(image):
    image     = preprocess_for_tesseract(image)
    pil_image = Image.fromarray(image)

    custom_config = r'--oem 3 --psm 6 -l ind+eng --dpi 300 -c tessedit_char_blacklist=~`!@#$%^&*_=+[]{}|;:"<>'

    text = pytesseract.image_to_string(pil_image, config=custom_config)
    return text