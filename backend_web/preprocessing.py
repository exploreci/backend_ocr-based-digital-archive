import cv2
import numpy as np
import imutils
from transform import four_point_transform


# ================================================================
# auto_canny
# Deteksi tepi otomatis dengan threshold berdasarkan median piksel
# ================================================================
def auto_canny(image):
    v     = np.median(image)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    return cv2.Canny(image, lower, upper)


# ================================================================
# get_document_contour
# Cari kontur segi-empat terbesar dari daftar kontur
# ================================================================
def get_document_contour(cnts):
    for c in cnts:
        peri  = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(c) > 1000:
            return approx
    return None


# ================================================================
# detect_document_corners
# Deteksi 4 sudut dokumen secara otomatis dari gambar
# Output: array (4,2) koordinat sudut, atau None jika gagal
# ================================================================
def detect_document_corners(image):
    resized = imutils.resize(image, height=500)
    ratio   = image.shape[0] / resized.shape[0]

    h, w = resized.shape[:2]
    x1, y1 = int(w * 0.1), int(h * 0.1)
    x2, y2 = int(w * 0.9), int(h * 0.9)
    roi = resized[y1:y2, x1:x2]

    gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray  = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = auto_canny(gray)

    kernel = np.ones((5, 5), np.uint8)
    edged  = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    screenCnt = get_document_contour(cnts)
    if screenCnt is None:
        return None

    screenCnt = screenCnt + [x1, y1]
    return screenCnt.reshape(4, 2) * ratio


# ================================================================
# enhance_for_pdf
# Perbaiki kualitas gambar: sharpening untuk PDF + grayscale untuk OCR
# Output: (img_color, img_ocr)
#   img_color → gambar berwarna tajam (untuk PDF arsip)
#   img_ocr   → gambar grayscale bersih (untuk Tesseract)
# ================================================================
def enhance_for_pdf(img):
    h, w = img.shape[:2]
    if w < 1500:
        scale = 1500 / w
        img   = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)

    gaussian    = cv2.GaussianBlur(img, (0, 0), 2)
    img_color   = cv2.addWeighted(img, 1.3, gaussian, -0.3, 0)

    img_ocr = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    img_ocr = cv2.normalize(img_ocr, None, 0, 255, cv2.NORM_MINMAX)

    return img_color, img_ocr


# ================================================================
# prepare_image_for_ocr
# Fungsi utama preprocessing:
#   1. Crop/warp dokumen (manual atau auto-detect)
#   2. Enhance kualitas gambar
# Input : image (numpy BGR), manual_points (opsional, array 4x2)
# Output: (img_color, img_ocr)
# ================================================================
def prepare_image_for_ocr(image, manual_points=None):
    if manual_points is not None and len(manual_points) == 4:
        print("[PREPROCESSING] Mode manual crop")
        pts    = np.array(manual_points, dtype="float32")
        warped = four_point_transform(image, pts)
    else:
        print("[PREPROCESSING] Mode auto detect")
        corners = detect_document_corners(image)

        if corners is None:
            print("[PREPROCESSING] Auto detect gagal, pakai full image")
            warped = image
        else:
            print("[PREPROCESSING] Auto detect sukses")
            warped = four_point_transform(image, corners)

    img_color, img_ocr = enhance_for_pdf(warped)
    return img_color, img_ocr

    