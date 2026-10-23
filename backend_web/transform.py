import json
import numpy as np
import cv2


# ================================================================
# order_points
# Urutkan 4 titik: top-left, top-right, bottom-right, bottom-left
# ================================================================
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect


# ================================================================
# four_point_transform
# Warp perspektif gambar berdasarkan 4 titik sudut dokumen
# ================================================================
def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA  = np.linalg.norm(br - bl)
    widthB  = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA  = np.linalg.norm(tr - br)
    heightB  = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M      = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


# ================================================================
# parse_manual_coords
# Konversi koordinat Flutter (layar) → koordinat piksel gambar
# Input : coords_json (string JSON), screen_w, screen_h, image
# Output: numpy array (4,2) float32 dalam skala piksel gambar
#         atau None jika input tidak valid
# ================================================================
def parse_manual_coords(coords_json, screen_w, screen_h, image):
    if not coords_json or not screen_w or not screen_h:
        return None

    try:
        pts     = np.array(json.loads(coords_json), dtype="float32")
        img_h, img_w = image.shape[:2]

        ratio_w = img_w / float(screen_w)
        ratio_h = img_h / float(screen_h)

        manual_pts = pts * [ratio_w, ratio_h]

        print(f"✔ [TRANSFORM] parse_manual_coords | ratio: {ratio_w:.2f}, {ratio_h:.2f}")
        return manual_pts

    except Exception as e:
        print(f"⚠ [TRANSFORM] parse_manual_coords gagal: {e}")
        return None