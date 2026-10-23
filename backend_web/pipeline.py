import io
import base64
import cv2
import numpy as np          # ← tambah ini
from PIL import Image
from typing import Optional

from transform      import parse_manual_coords
from preprocessing  import prepare_image_for_ocr
from ocr_engine     import extract_text
from rbs            import extract_metadata


# ================================================================
# decode_bytes_to_image
# Terima raw bytes (PDF atau gambar) → numpy array BGR
# Dipanggil oleh ocr_endpoint sebelum process_document
# ================================================================
def decode_bytes_to_image(content: bytes, filename: str):
    """
    Decode raw file bytes → numpy BGR image.
    Raise ValueError jika file corrupt atau tidak bisa dibaca.
    """
    if not content:
        raise ValueError("File kosong atau tidak dapat dibaca.")

    if filename.lower().endswith(".pdf"):
        try:
            import fitz
            pdf_doc   = fitz.open(stream=content, filetype="pdf")
            page      = pdf_doc[0]
            pix       = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
            img_bytes = pix.tobytes("png")
            pdf_doc.close()
            nparr = np.frombuffer(img_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            raise ValueError(f"Gagal membaca PDF: {str(e)}")
    else:
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None or image.size == 0:
        raise ValueError("File tidak dapat diproses. Pastikan file tidak rusak atau corrupt.")

    return image


# ================================================================
# process_document  (tidak berubah, cukup hapus validasi image 
# yang tadi ditambahkan karena sudah ditangani decode_bytes_to_image)
# ================================================================
def process_document(
    image,
    coords_json: Optional[str] = None,
    screen_w:    Optional[float] = None,
    screen_h:    Optional[float] = None,
    jenis_surat: str = "masuk",
    on_status=None,
) -> dict:

    def _update(s):
        if on_status:
            on_status(s)

    # --- STEP 1: Parse koordinat manual ---
    _update("parsing_coords")
    print("▶ [PIPELINE] Step 1: Parse koordinat...")
    manual_pts = parse_manual_coords(coords_json, screen_w, screen_h, image)

    # --- STEP 2: Preprocessing ---
    _update("preprocessing")
    print("▶ [PIPELINE] Step 2: Preprocessing (crop & enhance)...")
    img_color, img_ocr = prepare_image_for_ocr(image, manual_points=manual_pts)

    if img_color is None or img_ocr is None:
        raise ValueError("Dokumen tidak dapat diproses. Kualitas gambar terlalu rendah atau sudut tidak terdeteksi.")

    # --- STEP 3: OCR ---
    _update("ocr")
    print("▶ [PIPELINE] Step 3: Ekstraksi teks OCR...")
    raw_text = extract_text(img_ocr)

    # --- STEP 4: Generate PDF ---
    _update("generating_pdf")
    print("▶ [PIPELINE] Step 4: Generate PDF...")
    img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    pdf_io  = io.BytesIO()
    pil_img.save(pdf_io, format="PDF", resolution=100.0)
    pdf_base64 = base64.b64encode(pdf_io.getvalue()).decode()

    # --- STEP 5: Ekstraksi metadata RBS ---
    _update("extracting_metadata")
    print("▶ [PIPELINE] Step 5: Rule-Based System (metadata)...")
    metadata = extract_metadata(raw_text, jenis_surat=jenis_surat)

    print("✅ [PIPELINE] Selesai.")
    return {
        "success"   : True,
        "text"      : raw_text,
        "metadata"  : metadata,
        "pdf_base64": pdf_base64,
    }