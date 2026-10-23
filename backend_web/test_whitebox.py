import unittest
import sys
import os
import numpy as np
import cv2
import json

# ── Path agar bisa import file backend ──
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# ══════════════════════════════════════════════════════════════════════
# IMPORT MODUL YANG DIUJI
# Salin semua file .py ke folder yang sama dengan file ini
# ══════════════════════════════════════════════════════════════════════
from rbs import clean_noise, normalize_text, extract_metadata
from transform import order_points, four_point_transform
from preprocessing import (
    auto_canny, get_document_contour,
    enhance_for_pdf, prepare_image_for_ocr
)
from ocr_engine import preprocess_for_tesseract, extract_text
from pipeline import process_document


# ══════════════════════════════════════════════════════════════════════
# 1. PENGUJIAN: rbs.py → clean_noise()
# Jalur logika: input kosong, hapus noise, koreksi kata, normalisasi spasi
# ══════════════════════════════════════════════════════════════════════
class TestCleanNoise(unittest.TestCase):

    def test_TC01_input_kosong(self):
        """TC01: Input string kosong → return string kosong (path: if not text)"""
        self.assertEqual(clean_noise(""), "")

    def test_TC02_input_none(self):
        """TC02: Input None → return string kosong"""
        self.assertEqual(clean_noise(None), "")

    def test_TC03_hapus_karakter_noise(self):
        """TC03: Karakter ! @ # dihapus (path: regex sub noise)"""
        result = clean_noise("Hello! @World#")
        self.assertNotIn("!", result)
        self.assertNotIn("@", result)
        self.assertNotIn("#", result)

    def test_TC04_koreksi_unwra(self):
        """TC04: 'Unwra' dikoreksi menjadi 'UNPRA'"""
        self.assertIn("UNPRA", clean_noise("Unwra Prabumulih"))

    def test_TC05_koreksi_prabumulh(self):
        """TC05: 'Prabumulh' dikoreksi menjadi 'Prabumulih'"""
        self.assertIn("Prabumulih", clean_noise("Prabumulh"))

    def test_TC06_koreksi_nornor(self):
        """TC06: 'Nornor' dikoreksi menjadi 'Nomor'"""
        self.assertIn("Nomor", clean_noise("Nornor Surat"))

    def test_TC07_normalisasi_spasi_berlebih(self):
        """TC07: Spasi ganda dinormalisasi menjadi satu spasi"""
        result = clean_noise("kata   banyak    spasi")
        self.assertNotIn("  ", result)

    def test_TC08_teks_normal_tidak_berubah(self):
        """TC08: Teks normal tanpa noise dikembalikan apa adanya"""
        result = clean_noise("Universitas Prabumulih")
        self.assertEqual(result, "Universitas Prabumulih")


# ══════════════════════════════════════════════════════════════════════
# 2. PENGUJIAN: rbs.py → normalize_text()
# Jalur logika: split per baris, clean tiap baris, filter baris kosong
# ══════════════════════════════════════════════════════════════════════
class TestNormalizeText(unittest.TestCase):

    def test_TC09_baris_kosong_dihapus(self):
        """TC09: Baris kosong difilter dari hasil (path: filter list)"""
        result = normalize_text("baris1\n\n\nbaris2")
        self.assertEqual(len(result), 2)

    def test_TC10_tiap_baris_dibersihkan(self):
        """TC10: Setiap baris diproses oleh clean_noise"""
        result = normalize_text("Unwra\nPrabumulh")
        self.assertIn("UNPRA", result[0])
        self.assertIn("Prabumulih", result[1])

    def test_TC11_input_kosong(self):
        """TC11: Input string kosong menghasilkan list kosong"""
        result = normalize_text("")
        self.assertEqual(result, [])

    def test_TC12_satu_baris(self):
        """TC12: Input satu baris menghasilkan list dengan 1 elemen"""
        result = normalize_text("Universitas Prabumulih")
        self.assertEqual(len(result), 1)


# ══════════════════════════════════════════════════════════════════════
# 3. PENGUJIAN: rbs.py → extract_metadata()
# Jalur logika: nomor surat, tanggal, perihal, asal/tujuan, fallback
# ══════════════════════════════════════════════════════════════════════
class TestExtractMetadata(unittest.TestCase):

    # ── Nomor Surat ─────────────────────────────────────────────────

    def test_TC13_nomor_surat_format_standar(self):
        """TC13: Nomor surat format standar ditemukan (path: regex match utama)"""
        text = "Nomor : 097/UNPRA/IV/2026"
        result = extract_metadata(text, "keluar")
        self.assertEqual(result["nomor_surat"], "097/UNPRA/IV/2026")

    def test_TC14_nomor_surat_typo_keyword(self):
        """TC14: Keyword 'Nornor' tetap bisa ditemukan (path: pola_keyword regex)"""
        text = "Nornor : 010/UNPRA/III/2025"
        result = extract_metadata(text, "keluar")
        self.assertEqual(result["nomor_surat"], "010/UNPRA/III/2025")

    def test_TC15_nomor_skip_baris_telp(self):
        """TC15: Baris mengandung 'telp' di-skip (path: if any noise → continue)"""
        text = "Telp: 081288764755\nNomor: 012/DINAS/V/2025"
        result = extract_metadata(text, "masuk")
        self.assertEqual(result["nomor_surat"], "012/DINAS/V/2025")

    def test_TC16_nomor_surat_fallback(self):
        """TC16: Fallback regex aktif jika keyword tidak ada"""
        text = "Surat ini bernomor 005/UNPRA/I/2026 perihal rapat"
        result = extract_metadata(text, "masuk")
        self.assertIn("/", result["nomor_surat"])

    def test_TC17_nomor_surat_tidak_ada(self):
        """TC17: Nomor surat tidak ditemukan → string kosong"""
        text = "Surat tanpa nomor sama sekali hanya teks biasa"
        result = extract_metadata(text, "masuk")
        self.assertEqual(result["nomor_surat"], "")

    # ── Tanggal Surat ────────────────────────────────────────────────

    def test_TC18_tanggal_format_indonesia(self):
        """TC18: Tanggal format Indonesia dikenali dan dikonversi ke YYYY-MM-DD"""
        text = "Prabumulih, 7 Mei 2026"
        result = extract_metadata(text, "keluar")
        self.assertEqual(result["tanggal_surat"], "2026-05-07")

    def test_TC19_tanggal_hari_dua_digit(self):
        """TC19: Hari dua digit diproses dengan benar"""
        text = "15 Januari 2025"
        result = extract_metadata(text, "masuk")
        self.assertEqual(result["tanggal_surat"], "2025-01-15")

    def test_TC20_tanggal_tidak_ada(self):
        """TC20: Tanggal tidak ditemukan → string kosong"""
        text = "Surat tanpa tanggal apapun"
        result = extract_metadata(text, "masuk")
        self.assertEqual(result["tanggal_surat"], "")

    def test_TC21_semua_bulan_dikenali(self):
        """TC21: Semua nama bulan Indonesia dikenali"""
        bulan_test = [
            ("Februari", "02"), ("Maret", "03"), ("April", "04"),
            ("Juni", "06"), ("Agustus", "08"), ("Oktober", "10"),
            ("Desember", "12"),
        ]
        for nama_bulan, kode in bulan_test:
            text = f"1 {nama_bulan} 2025"
            result = extract_metadata(text, "masuk")
            self.assertIn(kode, result["tanggal_surat"],
                          msg=f"Bulan {nama_bulan} tidak dikenali")

    # ── Perihal ──────────────────────────────────────────────────────

    def test_TC22_perihal_keyword_perihal(self):
        """TC22: Perihal dengan keyword 'Perihal' ditemukan"""
        text = "Perihal : Undangan Rapat Koordinasi"
        result = extract_metadata(text, "keluar")
        self.assertIn("Undangan", result["perihal"])

    def test_TC23_perihal_keyword_hal(self):
        """TC23: Perihal dengan keyword 'Hal' ditemukan (path: alternatif keyword)"""
        text = "Hal : Permohonan Izin Penelitian"
        result = extract_metadata(text, "masuk")
        self.assertIn("Permohonan", result["perihal"])

    def test_TC24_perihal_teks_pendek_lanjut_baris(self):
        """TC24: Isi perihal pendek → ambil baris berikutnya (path: len(isi) < 5)"""
        text = "Perihal :\nUndangan Rapat Koordinasi Semester Genap"
        result = extract_metadata(text, "keluar")
        self.assertIn("Undangan", result["perihal"])

    def test_TC25_perihal_tidak_ada(self):
        """TC25: Perihal tidak ditemukan → string kosong"""
        text = "Nomor: 001/X/2025\nTanggal: 1 Januari 2025"
        result = extract_metadata(text, "masuk")
        self.assertEqual(result["perihal"], "")

    # ── Asal Surat (masuk) ───────────────────────────────────────────

    def test_TC26_asal_dari_kop_universitas(self):
        """TC26: Asal surat dari kop dengan keyword UNIVERSITAS"""
        text = "UNIVERSITAS PRABUMULIH\nJalan Patra No.50"
        result = extract_metadata(text, "masuk")
        self.assertIn("UNIVERSITAS", result["asal_surat"].upper())

    def test_TC27_asal_dari_kop_dinas(self):
        """TC27: Asal surat dari kop dengan keyword DINAS"""
        text = "DINAS PENDIDIKAN KOTA PRABUMULIH\nJalan Sudirman No.1"
        result = extract_metadata(text, "masuk")
        self.assertIn("DINAS", result["asal_surat"].upper())

    def test_TC28_asal_fallback_baris_pertama(self):
        """TC28: Tidak ada keyword kop → ambil baris pertama (path: fallback)"""
        text = "Lembaga Swasta Mandiri\nNomor: 001/LSM/I/2025"
        result = extract_metadata(text, "masuk")
        self.assertEqual(result["asal_surat"], "Lembaga Swasta Mandiri")

    # ── Tujuan Surat (keluar) ────────────────────────────────────────

    def test_TC29_tujuan_kepada_yth(self):
        """TC29: Tujuan dari 'Kepada Yth.' ditemukan"""
        text = "Kepada Yth.\nRektor Universitas Prabumulih"
        result = extract_metadata(text, "keluar")
        self.assertIn("Rektor", result["tujuan_surat"])

    def test_TC30_tujuan_multi_list_bernomor(self):
        """TC30: Tujuan list bernomor digabung dengan '|' (path: match_nomor)"""
        text = "Kepada Yth.\n1. Dekan Fakultas Teknik\n2. Dekan Fakultas Ekonomi"
        result = extract_metadata(text, "keluar")
        self.assertIn("|", result["tujuan_surat"])

    def test_TC31_tujuan_format_dash(self):
        """TC31: Tujuan format dash/bullet dikenali (path: match_dash)"""
        text = "Kepada Yth.\n- Kepala Biro Akademik\n- Kepala Biro Keuangan"
        result = extract_metadata(text, "keluar")
        self.assertIn("|", result["tujuan_surat"])

    def test_TC32_tujuan_tidak_ada(self):
        """TC32: Tidak ada penerima → tujuan_surat kosong"""
        text = "Nomor: 001/X/2025\nPerihal: Undangan"
        result = extract_metadata(text, "keluar")
        self.assertEqual(result["tujuan_surat"], "")

    def test_TC33_stop_saat_ketemu_keyword_lain(self):
        """TC33: Scan tujuan berhenti saat ketemu keyword 'Nomor'"""
        text = "Kepada Yth.\nRektor UNPRA\nNomor: 001/X/2025\nDekan FT"
        result = extract_metadata(text, "keluar")
        # "Dekan FT" tidak boleh masuk tujuan karena setelah "Nomor"
        self.assertNotIn("Dekan FT", result["tujuan_surat"])

    # ── Edge Case ────────────────────────────────────────────────────

    def test_TC34_input_kosong(self):
        """TC34: Input teks kosong → semua field kosong"""
        result = extract_metadata("", "masuk")
        self.assertEqual(result["nomor_surat"], "")
        self.assertEqual(result["perihal"], "")
        self.assertEqual(result["tanggal_surat"], "")

    def test_TC35_jenis_surat_none(self):
        """TC35: jenis_surat=None → asal & tujuan tidak diisi, field lain tetap jalan"""
        text = "Nomor: 005/X/2025\nPerihal: Test\n7 Mei 2026"
        result = extract_metadata(text, None)
        self.assertEqual(result["asal_surat"], "")
        self.assertEqual(result["tujuan_surat"], "")
        self.assertNotEqual(result["nomor_surat"], "")


# ══════════════════════════════════════════════════════════════════════
# 4. PENGUJIAN: transform.py → order_points()
# Jalur logika: pengurutan 4 titik (TL, TR, BR, BL)
# ══════════════════════════════════════════════════════════════════════
class TestOrderPoints(unittest.TestCase):

    def _buat_pts(self, tl, tr, br, bl):
        """Helper: buat array 4 titik dalam urutan acak"""
        return np.array([tl, br, tr, bl], dtype="float32")

    def test_TC36_urutan_titik_benar(self):
        """TC36: 4 titik diurutkan TL-TR-BR-BL dengan benar"""
        pts = self._buat_pts(
            tl=(0, 0), tr=(100, 0),
            br=(100, 150), bl=(0, 150)
        )
        rect = order_points(pts)
        # TL = sum terkecil (0+0=0)
        np.testing.assert_array_equal(rect[0], [0, 0])
        # BR = sum terbesar (100+150=250)
        np.testing.assert_array_equal(rect[2], [100, 150])

    def test_TC37_output_shape_benar(self):
        """TC37: Output order_points selalu shape (4,2)"""
        pts = np.array([[0,0],[100,0],[100,100],[0,100]], dtype="float32")
        rect = order_points(pts)
        self.assertEqual(rect.shape, (4, 2))

    def test_TC38_titik_tidak_simetris(self):
        """TC38: Titik tidak simetris tetap diurutkan dengan benar"""
        pts = np.array([[10,5],[200,3],[195,300],[5,290]], dtype="float32")
        rect = order_points(pts)
        # TL harus punya nilai x+y terkecil
        sums = [p[0]+p[1] for p in pts]
        self.assertEqual(rect[0][0]+rect[0][1], min(sums))


# ══════════════════════════════════════════════════════════════════════
# 5. PENGUJIAN: transform.py → four_point_transform()
# Jalur logika: warp perspektif gambar ke bentuk persegi panjang
# ══════════════════════════════════════════════════════════════════════
class TestFourPointTransform(unittest.TestCase):

    def _buat_gambar(self, h=300, w=200):
        """Helper: buat gambar dummy BGR"""
        return np.zeros((h, w, 3), dtype=np.uint8)

    def test_TC39_output_adalah_numpy_array(self):
        """TC39: Output four_point_transform adalah numpy array"""
        img = self._buat_gambar()
        pts = np.array([[0,0],[200,0],[200,300],[0,300]], dtype="float32")
        result = four_point_transform(img, pts)
        self.assertIsInstance(result, np.ndarray)

    def test_TC40_output_tidak_kosong(self):
        """TC40: Output tidak kosong (height dan width > 0)"""
        img = self._buat_gambar()
        pts = np.array([[0,0],[199,0],[199,299],[0,299]], dtype="float32")
        result = four_point_transform(img, pts)
        self.assertGreater(result.shape[0], 0)
        self.assertGreater(result.shape[1], 0)

    def test_TC41_transformasi_persegi_panjang(self):
        """TC41: Transformasi persegi → output mendekati ukuran yang sama"""
        img = self._buat_gambar(400, 300)
        pts = np.array([[0,0],[299,0],[299,399],[0,399]], dtype="float32")
        result = four_point_transform(img, pts)
        # Lebar output sekitar 300, tinggi sekitar 400 (toleransi ±2)
        self.assertAlmostEqual(result.shape[1], 299, delta=2)
        self.assertAlmostEqual(result.shape[0], 399, delta=2)

    def test_TC42_input_titik_miring(self):
        """TC42: Input titik miring (perspektif) tetap menghasilkan output"""
        img = self._buat_gambar(400, 300)
        # Simulasi dokumen miring
        pts = np.array([[30,10],[280,0],[300,390],[0,400]], dtype="float32")
        result = four_point_transform(img, pts)
        self.assertIsInstance(result, np.ndarray)
        self.assertGreater(result.size, 0)


# ══════════════════════════════════════════════════════════════════════
# 6. PENGUJIAN: preprocessing.py → auto_canny()
# Jalur logika: hitung median → lower/upper threshold → cv2.Canny
# ══════════════════════════════════════════════════════════════════════
class TestAutoCanny(unittest.TestCase):

    def test_TC43_output_shape_sama_input(self):
        """TC43: Shape output sama dengan input (path: Canny dijalankan)"""
        gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = auto_canny(gray)
        self.assertEqual(result.shape, gray.shape)

    def test_TC44_output_adalah_edge_map(self):
        """TC44: Output hanya berisi nilai 0 atau 255 (binary edge map)"""
        gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = auto_canny(gray)
        unique_vals = set(np.unique(result).tolist())
        self.assertTrue(unique_vals.issubset({0, 255}))

    def test_TC45_gambar_seragam_no_edge(self):
        """TC45: Gambar seragam satu warna → tidak ada edge (semua 0)"""
        gray = np.ones((100, 100), dtype=np.uint8) * 128
        result = auto_canny(gray)
        self.assertEqual(np.sum(result), 0)

    def test_TC46_gambar_hitam_lower_upper_valid(self):
        """TC46: Gambar hitam (median=0) → lower=0, upper=0, tidak crash"""
        gray = np.zeros((100, 100), dtype=np.uint8)
        try:
            result = auto_canny(gray)
            self.assertIsInstance(result, np.ndarray)
        except Exception as e:
            self.fail(f"auto_canny crash pada gambar hitam: {e}")

    def test_TC47_gambar_putih_lower_upper_valid(self):
        """TC47: Gambar putih (median=255) → lower/upper di-clamp ke 255, tidak crash"""
        gray = np.ones((100, 100), dtype=np.uint8) * 255
        try:
            result = auto_canny(gray)
            self.assertIsInstance(result, np.ndarray)
        except Exception as e:
            self.fail(f"auto_canny crash pada gambar putih: {e}")


# ══════════════════════════════════════════════════════════════════════
# 7. PENGUJIAN: preprocessing.py → get_document_contour()
# Jalur logika: loop kontur → cek 4 titik → cek area → return / return None
# ══════════════════════════════════════════════════════════════════════
class TestGetDocumentContour(unittest.TestCase):

    def _buat_contour_4_titik(self, area_besar=True):
        """Helper: buat kontur 4 titik dengan area besar atau kecil"""
        if area_besar:
            pts = np.array([[[10,10]],[[200,10]],[[200,200]],[[10,200]]], dtype=np.int32)
        else:
            pts = np.array([[[0,0]],[[5,0]],[[5,5]],[[0,5]]], dtype=np.int32)
        return pts

    def test_TC48_kontur_4_titik_area_besar_ditemukan(self):
        """TC48: Kontur 4 titik area besar → dikembalikan (path: len==4 dan area>1000)"""
        cnts = [self._buat_contour_4_titik(area_besar=True)]
        result = get_document_contour(cnts)
        self.assertIsNotNone(result)

    def test_TC49_kontur_4_titik_area_kecil_dilewati(self):
        """TC49: Kontur 4 titik area kecil (<1000) → return None (path: area <= 1000)"""
        cnts = [self._buat_contour_4_titik(area_besar=False)]
        result = get_document_contour(cnts)
        self.assertIsNone(result)

    def test_TC50_kontur_bukan_4_titik_dilewati(self):
        """TC50: Kontur bukan 4 titik (segitiga) → return None (path: len != 4)"""
        segitiga = np.array([[[0,0]],[[100,0]],[[50,100]]], dtype=np.int32)
        result = get_document_contour([segitiga])
        self.assertIsNone(result)

    def test_TC51_list_kosong_return_none(self):
        """TC51: List kontur kosong → return None (path: loop tidak berjalan)"""
        result = get_document_contour([])
        self.assertIsNone(result)


# ══════════════════════════════════════════════════════════════════════
# 8. PENGUJIAN: preprocessing.py → enhance_for_pdf()
# Jalur logika: resize jika perlu → unsharp mask → grayscale → normalize
# ══════════════════════════════════════════════════════════════════════
class TestEnhanceForPdf(unittest.TestCase):

    def test_TC52_output_dua_gambar(self):
        """TC52: Fungsi mengembalikan tuple 2 elemen (img_color, img_ocr)"""
        img = np.ones((300, 200, 3), dtype=np.uint8) * 128
        result = enhance_for_pdf(img)
        self.assertEqual(len(result), 2)

    def test_TC53_img_color_adalah_bgr(self):
        """TC53: img_color memiliki 3 channel (BGR/warna)"""
        img = np.ones((300, 200, 3), dtype=np.uint8) * 128
        img_color, _ = enhance_for_pdf(img)
        self.assertEqual(len(img_color.shape), 3)
        self.assertEqual(img_color.shape[2], 3)

    def test_TC54_img_ocr_adalah_grayscale(self):
        """TC54: img_ocr memiliki 1 channel (grayscale untuk Tesseract)"""
        img = np.ones((300, 200, 3), dtype=np.uint8) * 128
        _, img_ocr = enhance_for_pdf(img)
        self.assertEqual(len(img_ocr.shape), 2)

    def test_TC55_gambar_kecil_di_upscale(self):
        """TC55: Gambar lebar < 1500px → di-upscale (path: if w < 1500)"""
        img = np.ones((300, 200, 3), dtype=np.uint8) * 128  # lebar 200 < 1500
        img_color, _ = enhance_for_pdf(img)
        self.assertGreaterEqual(img_color.shape[1], 1500)

    def test_TC56_gambar_besar_tidak_di_upscale(self):
        """TC56: Gambar lebar >= 1500px → tidak di-upscale (path: if w >= 1500 skip)"""
        img = np.ones((300, 1600, 3), dtype=np.uint8) * 128  # lebar 1600 >= 1500
        img_color, _ = enhance_for_pdf(img)
        self.assertEqual(img_color.shape[1], 1600)

    def test_TC57_img_ocr_nilai_0_sampai_255(self):
        """TC57: Nilai pixel img_ocr dalam range 0-255 setelah normalisasi"""
        img = np.random.randint(50, 200, (300, 200, 3), dtype=np.uint8)
        _, img_ocr = enhance_for_pdf(img)
        self.assertGreaterEqual(int(img_ocr.min()), 0)
        self.assertLessEqual(int(img_ocr.max()), 255)


# ══════════════════════════════════════════════════════════════════════
# 9. PENGUJIAN: preprocessing.py → prepare_image_for_ocr()
# Jalur logika: manual_points → warp, atau auto detect → fallback full image
# ══════════════════════════════════════════════════════════════════════
class TestPrepareImageForOcr(unittest.TestCase):

    def _buat_gambar(self, h=400, w=300):
        return np.ones((h, w, 3), dtype=np.uint8) * 200

    def test_TC58_manual_points_valid_4_titik(self):
        """TC58: manual_points 4 titik → pakai four_point_transform (path: manual mode)"""
        img = self._buat_gambar()
        pts = np.array([[0,0],[299,0],[299,399],[0,399]], dtype="float32")
        img_color, img_ocr = prepare_image_for_ocr(img, manual_points=pts)
        self.assertIsInstance(img_color, np.ndarray)
        self.assertIsInstance(img_ocr, np.ndarray)

    def test_TC59_manual_points_none_pakai_auto(self):
        """TC59: manual_points=None → jalur auto detect dijalankan"""
        img = self._buat_gambar()
        try:
            img_color, img_ocr = prepare_image_for_ocr(img, manual_points=None)
            self.assertIsInstance(img_color, np.ndarray)
        except Exception as e:
            self.fail(f"prepare_image_for_ocr crash: {e}")

    def test_TC60_output_selalu_dua_array(self):
        """TC60: Output selalu tuple 2 numpy array apapun inputnya"""
        img = self._buat_gambar()
        result = prepare_image_for_ocr(img, manual_points=None)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], np.ndarray)
        self.assertIsInstance(result[1], np.ndarray)

    def test_TC61_manual_points_kurang_dari_4_pakai_auto(self):
        """TC61: manual_points < 4 titik → jatuh ke jalur auto (path: len != 4)"""
        img = self._buat_gambar()
        pts = np.array([[0,0],[100,0]], dtype="float32")  # hanya 2 titik
        try:
            img_color, img_ocr = prepare_image_for_ocr(img, manual_points=pts)
            self.assertIsInstance(img_color, np.ndarray)
        except Exception as e:
            self.fail(f"Crash dengan manual_points < 4: {e}")


# ══════════════════════════════════════════════════════════════════════
# 10. PENGUJIAN: ocr_engine.py → preprocess_for_tesseract()
# Jalur logika: cek shape → convert BGR→gray (jika perlu) → adaptiveThreshold
# ══════════════════════════════════════════════════════════════════════
class TestPreprocessForTesseract(unittest.TestCase):

    def test_TC62_input_bgr_dikonversi_grayscale(self):
        """TC62: Input BGR (3 channel) → output 2D grayscale (path: len==3 → cvtColor)"""
        bgr = np.ones((100, 100, 3), dtype=np.uint8) * 200
        result = preprocess_for_tesseract(bgr)
        self.assertEqual(len(result.shape), 2)

    def test_TC63_input_sudah_grayscale_tidak_convert(self):
        """TC63: Input grayscale (2D) → tidak dikonversi, langsung threshold (path: len!=3)"""
        gray = np.ones((100, 100), dtype=np.uint8) * 128
        result = preprocess_for_tesseract(gray)
        self.assertEqual(len(result.shape), 2)

    def test_TC64_output_hanya_hitam_putih(self):
        """TC64: Setelah adaptiveThreshold, pixel hanya 0 atau 255"""
        bgr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = preprocess_for_tesseract(bgr)
        unique_vals = set(np.unique(result).tolist())
        self.assertTrue(unique_vals.issubset({0, 255}))

    def test_TC65_output_shape_sama_dengan_input(self):
        """TC65: Ukuran output (h,w) sama dengan input"""
        bgr = np.ones((200, 150, 3), dtype=np.uint8) * 100
        result = preprocess_for_tesseract(bgr)
        self.assertEqual(result.shape, (200, 150))

    def test_TC66_output_bertipe_numpy_uint8(self):
        """TC66: Output bertipe numpy array uint8"""
        bgr = np.ones((100, 100, 3), dtype=np.uint8) * 128
        result = preprocess_for_tesseract(bgr)
        self.assertEqual(result.dtype, np.uint8)


# ══════════════════════════════════════════════════════════════════════
# 11. PENGUJIAN: ocr_engine.py → extract_text()
# Jalur logika: preprocess → PIL convert → pytesseract → return string
# ══════════════════════════════════════════════════════════════════════
class TestExtractText(unittest.TestCase):

    def test_TC67_output_bertipe_string(self):
        """TC67: Output extract_text selalu bertipe string"""
        img = np.ones((200, 200, 3), dtype=np.uint8) * 255
        result = extract_text(img)
        self.assertIsInstance(result, str)

    def test_TC68_gambar_putih_teks_kosong(self):
        """TC68: Gambar putih kosong → teks kosong/whitespace (path: Tesseract tidak menemukan teks)"""
        img = np.ones((200, 200, 3), dtype=np.uint8) * 255
        result = extract_text(img)
        self.assertEqual(result.strip(), "")

    def test_TC69_gambar_hitam_tidak_crash(self):
        """TC69: Gambar hitam tidak menyebabkan crash (path: tetap jalan, return string)"""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        try:
            result = extract_text(img)
            self.assertIsInstance(result, str)
        except Exception as e:
            self.fail(f"extract_text crash pada gambar hitam: {e}")

    def test_TC70_input_grayscale_tidak_crash(self):
        """TC70: Input 2D grayscale juga bisa diproses tanpa crash"""
        gray = np.ones((200, 200), dtype=np.uint8) * 200
        try:
            result = extract_text(gray)
            self.assertIsInstance(result, str)
        except Exception as e:
            self.fail(f"extract_text crash pada input grayscale: {e}")


# ══════════════════════════════════════════════════════════════════════
# 12. PENGUJIAN: pipeline.py → process_document()
# Jalur logika: parse koordinat → preprocessing → OCR → PDF → return dict
# ══════════════════════════════════════════════════════════════════════
class TestProcessDocument(unittest.TestCase):

    def _buat_gambar(self, h=400, w=300):
        """Helper: buat gambar dummy BGR"""
        return np.ones((h, w, 3), dtype=np.uint8) * 200

    def test_TC71_output_memiliki_key_wajib(self):
        """TC71: Output berisi key: success, text, pdf_base64, metadata"""
        img = self._buat_gambar()
        result = process_document(img)
        self.assertIn("success", result)
        self.assertIn("text", result)
        self.assertIn("pdf_base64", result)
        self.assertIn("metadata", result)

    def test_TC72_success_selalu_true(self):
        """TC72: Field 'success' selalu True jika tidak ada exception"""
        img = self._buat_gambar()
        result = process_document(img)
        self.assertTrue(result["success"])

    def test_TC73_text_bertipe_string(self):
        """TC73: Field 'text' bertipe string"""
        img = self._buat_gambar()
        result = process_document(img)
        self.assertIsInstance(result["text"], str)

    def test_TC74_pdf_base64_bertipe_string(self):
        """TC74: Field 'pdf_base64' bertipe string (base64 encoded)"""
        img = self._buat_gambar()
        result = process_document(img)
        self.assertIsInstance(result["pdf_base64"], str)

    def test_TC75_pdf_base64_tidak_kosong(self):
        """TC75: Field 'pdf_base64' tidak kosong (PDF berhasil di-generate)"""
        img = self._buat_gambar()
        result = process_document(img)
        self.assertGreater(len(result["pdf_base64"]), 0)

    def test_TC76_metadata_memiliki_key_lengkap(self):
        """TC76: Metadata memiliki semua 4 key yang dibutuhkan frontend"""
        img = self._buat_gambar()
        result = process_document(img)
        meta = result["metadata"]
        self.assertIn("nomor_surat", meta)
        self.assertIn("tanggal_surat", meta)
        self.assertIn("perihal", meta)
        self.assertIn("asal_surat", meta)

    def test_TC77_tanpa_koordinat_tetap_jalan(self):
        """TC77: coords_json=None → jalur manual dilewati, tetap jalan (path: if coords_json → skip)"""
        img = self._buat_gambar()
        result = process_document(img, coords_json=None, screen_w=None, screen_h=None)
        self.assertTrue(result["success"])

    def test_TC78_koordinat_valid_diproses(self):
        """TC78: coords_json valid + screen_w/h → manual_pts dihitung (path: parse koordinat)"""
        img = self._buat_gambar(400, 300)
        coords = json.dumps([[0,0],[300,0],[300,400],[0,400]])
        result = process_document(img, coords_json=coords, screen_w=300, screen_h=400)
        self.assertTrue(result["success"])

    def test_TC79_koordinat_json_rusak_tidak_crash(self):
        """TC79: coords_json tidak valid → exception ditangkap, pipeline lanjut (path: except → print warning)"""
        img = self._buat_gambar()
        result = process_document(img, coords_json="BUKAN_JSON", screen_w=300, screen_h=400)
        self.assertTrue(result["success"])

    def test_TC80_output_bukan_none(self):
        """TC80: Fungsi tidak pernah return None"""
        img = self._buat_gambar()
        result = process_document(img)
        self.assertIsNotNone(result)


# ══════════════════════════════════════════════════════════════════════
# JALANKAN SEMUA TEST
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    unittest.main(verbosity=2)