import os
import re
import io
import uuid
import time
import numpy as np
import cv2
from PIL import Image
from threading import Thread
from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel
from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import text

from config.db import get_db
from pipeline import process_document, decode_bytes_to_image


# ================================================================
# KONSTANTA VALIDASI FILE
# ================================================================
_MAX_FILE_SIZE   = 3 * 1024 * 1024
_ALLOWED_EXT     = {".pdf", ".jpg", ".jpeg", ".png"}
_ALLOWED_EXT_STR = "PDF, JPG, JPEG, PNG"

router = APIRouter()


# ================================================================
# IN-MEMORY JOB STORE
# Format: { job_id: { "status": str, "result": dict|None, "created_at": float } }
# Lifecycle: queued → processing → done | error
# ================================================================
_jobs: dict = {}


def _cleanup_old_jobs():
    """Hapus job yang sudah lebih dari 1 jam agar tidak bocor memori."""
    now     = time.time()
    expired = [jid for jid, data in _jobs.items() if now - data.get("created_at", now) > 3600]
    for jid in expired:
        del _jobs[jid]


# ================================================================
# POST /api/ocr
# ================================================================
@router.post("/api/ocr")
async def ocr_endpoint(
    file:        UploadFile      = File(...),
    coords:      Optional[str]  = Form(None),
    width:       Optional[float] = Form(None),
    height:      Optional[float] = Form(None),
    jenis_surat: Optional[str]  = Form("masuk"),
):
    try:
        content  = await file.read()
        filename = file.filename.lower()

        ext = os.path.splitext(filename)[1]
        if ext not in _ALLOWED_EXT:
            raise HTTPException(400, f"Tipe file tidak didukung. Gunakan: {_ALLOWED_EXT_STR}.")
        if len(content) > _MAX_FILE_SIZE:
            raise HTTPException(400, f"Ukuran file terlalu besar. Maksimal {_MAX_FILE_SIZE // (1024 * 1024)}MB.")

        try:
            image = decode_bytes_to_image(content, filename)
        except ValueError as e:
            raise HTTPException(400, str(e))

        job_id        = str(uuid.uuid4())
        _jobs[job_id] = {"status": "queued", "result": None, "created_at": time.time()}

        def _worker():
            try:
                result = process_document(
                    image, coords, width, height, jenis_surat,
                    on_status=lambda s: _jobs[job_id].update({"status": s})
                )
                _jobs[job_id]["status"] = "done"
                _jobs[job_id]["result"] = result
            except ValueError as e:
                _jobs[job_id]["status"]     = "error"
                _jobs[job_id]["error_type"] = "unprocessable"
                _jobs[job_id]["message"]    = str(e)
            except Exception as e:
                _jobs[job_id]["status"]     = "error"
                _jobs[job_id]["error_type"] = "server_error"
                _jobs[job_id]["message"]    = str(e)

        Thread(target=_worker, daemon=True).start()
        _cleanup_old_jobs()

        print(f"🚀 [JOB {job_id}] Diterima & antri diproses.")
        return {"success": True, "job_id": job_id}

    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR OCR ENDPOINT: {str(e)}")
        raise HTTPException(500, f"Terjadi kesalahan: {str(e)}")


# ================================================================
# GET /api/ocr/status/{job_id}
# ================================================================
@router.get("/api/ocr/status/{job_id}")
async def ocr_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job tidak ditemukan.")

    resp = {"status": job["status"]}

    if job["status"] == "error":
        resp["error_type"] = job.get("error_type", "server_error")
        resp["message"]    = job.get("message", "Terjadi kesalahan.")
    elif job["status"] == "done":
        resp.update(job["result"])

    return resp


# ================================================================
# POST /api/simpan-arsip
# ================================================================
@router.post("/api/simpan-arsip")
async def simpan_arsip(
    file:          UploadFile = File(...),
    nomor_surat:   str        = Form(...),
    tanggal_surat: str        = Form(...),
    perihal:       str        = Form(...),
    jenis_surat:   str        = Form(...),
    asal_tujuan:   str        = Form(...),
    id_user_input: int        = Form(...),
    id_unit:       int        = Form(...),
    db: Session = Depends(get_db),
):
    try:
        pdf_bytes = await file.read()

        ext = os.path.splitext(file.filename.lower())[1]
        if ext not in _ALLOWED_EXT:
            raise HTTPException(400, f"Tipe file tidak didukung. Gunakan: {_ALLOWED_EXT_STR}.")
        if len(pdf_bytes) > _MAX_FILE_SIZE:
            raise HTTPException(400, "Ukuran file terlalu besar. Maksimal 3MB.")

        cek_duplikat = db.execute(
            text("SELECT nomor_surat, tanggal_surat, perihal FROM surat WHERE nomor_surat = :nomor AND jenis_surat = :jenis LIMIT 1"),
            {"nomor": nomor_surat.strip(), "jenis": jenis_surat}
        ).fetchone()

        if cek_duplikat:
            return {
                "success"  : False,
                "duplikat" : True,
                "message"  : f"Nomor surat '{nomor_surat}' sudah tercatat di arsip {jenis_surat}.",
                "data_lama": {
                    "nomor_surat"  : cek_duplikat.nomor_surat,
                    "tanggal_surat": str(cek_duplikat.tanggal_surat),
                    "perihal"      : cek_duplikat.perihal,
                }
            }

        col_count    = "asal_surat" if jenis_surat == "masuk" else "tujuan_surat"
        result_count = db.execute(text(f"SELECT COUNT(*) FROM surat WHERE {col_count} = :val"), {"val": asal_tujuan}).scalar()

        prefix = "m" if jenis_surat == "masuk" else "k"
        urutan = (result_count or 0) + 1

        try:
            tgl_obj = datetime.strptime(tanggal_surat, "%Y-%m-%d")
        except Exception:
            tgl_obj = datetime.now()

        safe_asal_tujuan = re.sub(r'[^a-zA-Z0-9]', '', asal_tujuan.lower())
        filename         = f"{tgl_obj.strftime('%Y%m%d')}_{safe_asal_tujuan}_{prefix}{urutan}.pdf"
        folder_path      = "uploads/arsip"
        os.makedirs(folder_path, exist_ok=True)
        file_path_fisik  = os.path.join(folder_path, filename)

        with open(file_path_fisik, "wb") as f:
            f.write(pdf_bytes)

        db_path = file_path_fisik.replace("\\", "/")

        db.execute(
            text(f"""
                INSERT INTO surat (
                    nomor_surat, tanggal_surat, perihal, jenis_surat,
                    {col_count}, file_path, id_user_input, id_unit, created_at
                ) VALUES (:no, :tgl, :hal, :jenis, :asal, :path, :uid, :unit, NOW())
            """),
            {"no": nomor_surat, "tgl": tgl_obj.date(), "hal": perihal, "jenis": jenis_surat,
             "asal": asal_tujuan, "path": db_path, "uid": id_user_input, "unit": id_unit}
        )
        db.commit()

        return {"success": True, "message": "Arsip Universitas Prabumulih Berhasil Disimpan!", "filename": filename}

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"ERROR SIMPAN ARSIP: {str(e)}")
        return {"success": False, "message": f"Gagal masuk Database: {str(e)}"}


# ================================================================
# GET /api/surat-masuk
# ================================================================
@router.get("/api/surat-masuk")
async def get_surat_masuk(
    role:    str = Query(None),
    id_unit: int = Query(None),
    db: Session = Depends(get_db),
):
    try:
        sql_string = """
            SELECT s.*, COUNT(d.id_disposisi) as jml_disposisi
            FROM surat s
            LEFT JOIN disposisi d ON s.id_surat = d.id_surat
            WHERE s.jenis_surat = 'masuk'
        """
        params = {}

        if role and role.lower().strip() in ("operator", "pimpinan") and id_unit:
            sql_string += " AND s.id_unit = :id_unit"
            params["id_unit"] = id_unit

        sql_string += " GROUP BY s.id_surat ORDER BY s.created_at DESC"

        result     = db.execute(text(sql_string), params).mappings().all()
        final_data = []
        for row in result:
            new_row                 = dict(row)
            new_row['status_label'] = "Lihat Disposisi" if new_row['jml_disposisi'] > 0 else "Belum Disposisi"
            final_data.append(new_row)

        return {"success": True, "data": final_data}

    except Exception as e:
        return {"success": False, "message": str(e)}


# ================================================================
# GET /api/surat-keluar
# ================================================================
@router.get("/api/surat-keluar")
async def get_surat_keluar(
    role:    str = Query(None),
    id_unit: int = Query(None),
    db: Session = Depends(get_db),
):
    try:
        sql_string = "SELECT s.* FROM surat s WHERE s.jenis_surat = 'keluar'"
        params     = {}

        if role and role.lower().strip() in ("operator", "pimpinan") and id_unit:
            sql_string += " AND s.id_unit = :id_unit"
            params["id_unit"] = id_unit

        sql_string += " ORDER BY s.created_at DESC"

        result = db.execute(text(sql_string), params).mappings().all()
        return {"success": True, "data": [dict(row) for row in result]}

    except Exception as e:
        return {"success": False, "message": str(e)}


# ================================================================
# POST /api/disposisi
# Buat disposisi ke SATU atau BANYAK unit sekaligus
#
# Kolom tabel disposisi:
#   id_disposisi, id_surat, id_user, id_unit,
#   isi_disposisi, sifat_surat, status, tanggal_disposisi
# ================================================================
class DisposisiRequest(BaseModel):
    id_surat:         int
    id_unit_tujuan:   List[int]          # bisa kirim ke banyak unit
    isi_disposisi:    str
    id_user_pimpinan: int
    sifat_surat:      str                # 'Biasa' | 'Segera' | 'Rahasia'


@router.post("/api/disposisi")
async def buat_disposisi(data: DisposisiRequest, db: Session = Depends(get_db)):
    try:
        # --- Validasi input ---
        if not data.id_unit_tujuan:
            raise HTTPException(400, "Minimal pilih 1 unit tujuan disposisi.")

        allowed_sifat = {"Biasa", "Segera", "Rahasia"}
        if data.sifat_surat not in allowed_sifat:
            raise HTTPException(400, f"Sifat surat tidak valid. Gunakan: {', '.join(allowed_sifat)}.")

        # Ambil id_unit milik user pengirim (untuk cegah kirim ke unit sendiri)
        user_unit = db.execute(
            text("SELECT id_unit FROM users WHERE id_user = :id"),
            {"id": data.id_user_pimpinan}
        ).scalar()

        inserted = 0
        for id_unit in data.id_unit_tujuan:
            

            db.execute(
                text("""
                    INSERT INTO disposisi (
                        id_surat, id_user, id_unit,
                        isi_disposisi, sifat_surat,
                        status, tanggal_disposisi
                    ) VALUES (
                        :id_surat, :id_user, :id_unit,
                        :isi, :sifat,
                        'baru', :tgl
                    )
                """),
                {
                    "id_surat": data.id_surat,
                    "id_user" : data.id_user_pimpinan,
                    "id_unit" : id_unit,
                    "isi"     : data.isi_disposisi,
                    "sifat"   : data.sifat_surat,
                    "tgl"     : datetime.now(),
                }
            )
            inserted += 1

        db.commit()
        return {
            "success"        : True,
            "message"        : f"Disposisi berhasil dikirim ke {inserted} unit.",
            "jumlah_dikirim" : inserted,
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"ERROR SIMPAN DISPOSISI: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ================================================================
# GET /api/disposisi/terkirim
# Daftar disposisi yang dikirim oleh suatu unit (tab "Terkirim")
# Query param: unit_id (wajib) → id_unit dari user pengirim
# ================================================================
@router.get("/api/disposisi/terkirim")
async def get_disposisi_terkirim(
    unit_id: int = Query(...),
    db: Session = Depends(get_db),
):
    try:
        result = db.execute(
            text("""
                SELECT
                    d.id_disposisi,
                    d.id_surat,
                    d.id_user,
                    d.id_unit,
                    d.isi_disposisi,
                    d.sifat_surat,
                    d.status,
                    DATE_FORMAT(d.tanggal_disposisi, '%d-%m-%Y %H:%i') AS tanggal_disposisi,
                    s.nomor_surat,
                    s.perihal,
                    s.asal_surat,
                    DATE_FORMAT(s.tanggal_surat, '%d-%m-%Y') AS tanggal_surat,
                    u_ke.nama_unit  AS ke_unit,
                    us.nama_lengkap AS pengirim
                FROM disposisi d
                JOIN users  us   ON d.id_user  = us.id_user
                JOIN surat  s    ON d.id_surat = s.id_surat
                JOIN unit   u_ke ON d.id_unit  = u_ke.id_unit
                WHERE us.id_unit = :unit_id
                ORDER BY d.tanggal_disposisi DESC
            """),
            {"unit_id": unit_id}
        ).mappings().all()

        return {"success": True, "data": [dict(row) for row in result]}

    except Exception as e:
        print(f"ERROR GET DISPOSISI TERKIRIM: {str(e)}")
        return {"success": False, "message": str(e)}


# ================================================================
# GET /api/disposisi/masuk
# Daftar disposisi yang DITERIMA oleh suatu unit (tab "Disposisi Masuk")
# Query param: id_unit (wajib)
# ================================================================
@router.get("/api/disposisi/masuk")
async def get_disposisi_masuk(
    id_unit: int = Query(...),
    db: Session = Depends(get_db),
):
    try:
        result = db.execute(
            text("""
                SELECT
                    d.id_disposisi,
                    d.id_surat,
                    d.id_user,
                    d.id_unit,
                    d.isi_disposisi,
                    d.sifat_surat,
                    d.status,
                    DATE_FORMAT(d.tanggal_disposisi, '%d-%m-%Y %H:%i') AS tanggal_disposisi,
                    u_ke.nama_unit              AS ke_unit,
                    us.nama_lengkap             AS pengirim,
                    u_pengirim.nama_unit        AS dari_unit,
                    s.nomor_surat,
                    s.perihal,
                    s.asal_surat,
                    s.file_path,
                    DATE_FORMAT(s.tanggal_surat, '%d-%m-%Y') AS tanggal_surat
                FROM disposisi d
                JOIN users us           ON d.id_user  = us.id_user
                JOIN unit  u_pengirim   ON us.id_unit  = u_pengirim.id_unit
                JOIN unit  u_ke         ON d.id_unit   = u_ke.id_unit
                JOIN surat s            ON d.id_surat  = s.id_surat
                WHERE d.id_unit = :id_unit
                ORDER BY
                    FIELD(d.status, 'baru', 'dibaca', 'ditindaklanjuti'),
                    d.tanggal_disposisi DESC
            """),
            {"id_unit": id_unit}
        ).mappings().all()

        return {"success": True, "data": [dict(row) for row in result]}

    except Exception as e:
        print(f"ERROR GET DISPOSISI MASUK: {str(e)}")
        return {"success": False, "message": str(e)}


# ================================================================
# PATCH /api/disposisi/{id_disposisi}/status
# Update status disposisi oleh unit penerima
# Body: { "status": "dibaca" | "ditindaklanjuti" }
# ================================================================
class UpdateStatusRequest(BaseModel):
    status: str  # 'dibaca' | 'ditindaklanjuti'


@router.patch("/api/disposisi/{id_disposisi}/status")
async def update_status_disposisi(
    id_disposisi: int,
    data: UpdateStatusRequest,
    db: Session = Depends(get_db),
):
    try:
        allowed = {"dibaca", "ditindaklanjuti"}
        if data.status not in allowed:
            raise HTTPException(400, f"Status tidak valid. Gunakan: {', '.join(allowed)}.")

        cek = db.execute(
            text("SELECT id_disposisi FROM disposisi WHERE id_disposisi = :id"),
            {"id": id_disposisi}
        ).fetchone()

        if not cek:
            raise HTTPException(404, "Disposisi tidak ditemukan.")

        db.execute(
            text("UPDATE disposisi SET status = :status WHERE id_disposisi = :id"),
            {"status": data.status, "id": id_disposisi}
        )
        db.commit()

        return {"success": True, "message": f"Status disposisi diubah ke '{data.status}'."}

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"ERROR UPDATE STATUS DISPOSISI: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ================================================================
# GET /api/disposisi/{id_surat}
# Detail semua disposisi untuk 1 surat (untuk halaman detail surat)
# ================================================================
@router.get("/api/disposisi/{id_surat}")
async def get_disposisi(id_surat: int, db: Session = Depends(get_db)):
    try:
        result = db.execute(
            text("""
                SELECT
                    d.id_disposisi,
                    d.id_surat,
                    d.id_user,
                    d.id_unit,
                    d.isi_disposisi,
                    d.sifat_surat,
                    d.status,
                    DATE_FORMAT(d.tanggal_disposisi, '%d-%m-%Y %H:%i') AS tanggal_disposisi,
                    u_ke.nama_unit          AS ke_unit,
                    us.nama_lengkap         AS pengirim,
                    u_pengirim.nama_unit    AS dari_unit,
                    s.nomor_surat,
                    s.tanggal_surat,
                    s.asal_surat,
                    s.perihal
                FROM disposisi d
                JOIN users us           ON d.id_user  = us.id_user
                JOIN unit  u_pengirim   ON us.id_unit  = u_pengirim.id_unit
                JOIN unit  u_ke         ON d.id_unit   = u_ke.id_unit
                JOIN surat s            ON d.id_surat  = s.id_surat
                WHERE d.id_surat = :id_surat
                ORDER BY d.tanggal_disposisi DESC
            """),
            {"id_surat": id_surat}
        ).mappings().all()

        return {"success": True, "data": [dict(row) for row in result]}

    except Exception as e:
        print(f"ERROR GET DISPOSISI: {str(e)}")
        return {"success": False, "message": str(e)}


# ================================================================
# DELETE /api/disposisi/{id_disposisi}
# ================================================================
@router.delete("/api/disposisi/{id_disposisi}")
async def hapus_disposisi(id_disposisi: int, db: Session = Depends(get_db)):
    try:
        cek = db.execute(
            text("SELECT id_disposisi FROM disposisi WHERE id_disposisi = :id"),
            {"id": id_disposisi}
        ).fetchone()

        if not cek:
            raise HTTPException(status_code=404, detail="Disposisi tidak ditemukan")

        db.execute(text("DELETE FROM disposisi WHERE id_disposisi = :id"), {"id": id_disposisi})
        db.commit()
        return {"success": True, "message": "Instruksi disposisi berhasil dihapus"}

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


# ================================================================
# DELETE /api/surat/{id_surat}
# ================================================================
@router.delete("/api/surat/{id_surat}")
async def hapus_surat(id_surat: int, db: Session = Depends(get_db)):
    try:
        cek = db.execute(
            text("SELECT id_surat, file_path FROM surat WHERE id_surat = :id"),
            {"id": id_surat}
        ).fetchone()

        if not cek:
            raise HTTPException(status_code=404, detail="Surat tidak ditemukan")

        if cek.file_path and os.path.exists(cek.file_path):
            os.remove(cek.file_path)

        db.execute(text("DELETE FROM surat WHERE id_surat = :id"), {"id": id_surat})
        db.commit()
        return {"success": True, "message": "Arsip berhasil dihapus"}

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


# ================================================================
# PUT /api/surat/{id_surat}
# ================================================================
@router.put("/api/surat/{id_surat}")
async def edit_surat(id_surat: int, data: dict, db: Session = Depends(get_db)):
    try:
        cek = db.execute(
            text("SELECT id_surat, jenis_surat FROM surat WHERE id_surat = :id"),
            {"id": id_surat}
        ).fetchone()

        if not cek:
            raise HTTPException(status_code=404, detail="Surat tidak ditemukan")

        col_asal_tujuan = "asal_surat" if cek.jenis_surat == "masuk" else "tujuan_surat"
        val_asal_tujuan = data.get("asal_surat") or data.get("tujuan_surat", "")

        db.execute(
            text(f"""
                UPDATE surat SET
                    nomor_surat       = :no,
                    tanggal_surat     = :tgl,
                    perihal           = :hal,
                    {col_asal_tujuan} = :asal_tujuan
                WHERE id_surat = :id
            """),
            {"no": data.get("nomor_surat", ""), "tgl": data.get("tanggal_surat", ""),
             "hal": data.get("perihal", ""), "asal_tujuan": val_asal_tujuan, "id": id_surat}
        )
        db.commit()
        return {"success": True, "message": "Arsip berhasil diperbarui"}

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"ERROR EDIT SURAT: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ================================================================
# POST /api/simpan-surat-keluar
# ================================================================
class SuratKeluarRequest(BaseModel):
    nomor_surat:   str
    tanggal_surat: str
    perihal:       str
    tujuan_surat:  str
    id_user_input: int
    id_unit:       int


@router.post("/api/simpan-surat-keluar")
async def simpan_surat_keluar(data: SuratKeluarRequest, db: Session = Depends(get_db)):
    try:
        try:
            tgl_obj = datetime.strptime(data.tanggal_surat, "%Y-%m-%d")
        except Exception:
            tgl_obj = datetime.now()

        db.execute(
            text("""
                INSERT INTO surat (
                    nomor_surat, tanggal_surat, perihal, jenis_surat,
                    tujuan_surat, id_user_input, id_unit, created_at
                ) VALUES (:no, :tgl, :hal, 'keluar', :tujuan, :uid, :unit, NOW())
            """),
            {"no": data.nomor_surat, "tgl": tgl_obj.date(), "hal": data.perihal,
             "tujuan": data.tujuan_surat, "uid": data.id_user_input, "unit": data.id_unit}
        )
        db.commit()

        print(f"✅ Surat keluar '{data.nomor_surat}' berhasil disimpan.")
        return {"success": True, "message": "Metadata surat keluar berhasil disimpan!"}

    except Exception as e:
        db.rollback()
        print(f"ERROR SIMPAN SURAT KELUAR: {str(e)}")
        return {"success": False, "message": f"Gagal menyimpan: {str(e)}"}


# ================================================================
# PUT /api/surat/{id_surat}/upload-ttd
# ================================================================
@router.put("/api/surat/{id_surat}/upload-ttd")
async def upload_ttd(
    id_surat: int,
    file:     UploadFile = File(...),
    db: Session = Depends(get_db),
):
    try:
        cek = db.execute(
            text("SELECT id_surat, file_path FROM surat WHERE id_surat = :id"),
            {"id": id_surat}
        ).fetchone()

        if not cek:
            raise HTTPException(status_code=404, detail="Surat tidak ditemukan.")

        if cek.file_path and os.path.exists(cek.file_path):
            os.remove(cek.file_path)

        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in {".pdf", ".jpg", ".jpeg", ".png"}:
            raise HTTPException(status_code=400, detail="Format file tidak didukung. Gunakan PDF, JPG, atau PNG.")

        folder       = "uploads/arsip"
        os.makedirs(folder, exist_ok=True)
        new_filename = f"{uuid.uuid4().hex}_ttd{ext}"
        new_path     = os.path.join(folder, new_filename)

        raw_bytes = await file.read()
        with open(new_path, "wb") as f:
            f.write(raw_bytes)

        db_path = new_path.replace("\\", "/")

        db.execute(
            text("UPDATE surat SET file_path = :path WHERE id_surat = :id"),
            {"path": db_path, "id": id_surat}
        )
        db.commit()

        return {"success": True, "message": "PDF TTD berhasil diupload.", "file_path": db_path}

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"ERROR upload-ttd: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ================================================================
# GET /api/disposisi/masuk/jumlah
# Badge notifikasi: hitung disposisi baru yang belum dibaca
# Query param: id_unit (wajib)
# ================================================================
@router.get("/api/disposisi/masuk/jumlah")
async def get_jumlah_disposisi_masuk(
    id_unit: int = Query(...),
    db: Session = Depends(get_db),
):
    try:
        jumlah_baru = db.execute(
            text("""
                SELECT COUNT(*) FROM disposisi
                WHERE id_unit = :id_unit AND status = 'baru'
            """),
            {"id_unit": id_unit}
        ).scalar()

        return {"success": True, "jumlah_baru": int(jumlah_baru or 0)}

    except Exception as e:
        print(f"ERROR GET JUMLAH DISPOSISI MASUK: {str(e)}")
        return {"success": False, "message": str(e), "jumlah_baru": 0}


# ================================================================
# GET /api/unit
# Daftar semua unit (untuk dropdown disposisi)
# ================================================================
@router.get("/api/unit")
async def get_unit(db: Session = Depends(get_db)):
    try:
        result = db.execute(
            text("SELECT id_unit, nama_unit FROM unit ORDER BY nama_unit ASC")
        ).mappings().all()
        return {"success": True, "data": [dict(row) for row in result]}

    except Exception as e:
        print(f"ERROR GET UNIT: {str(e)}")
        return {"success": False, "message": str(e)}









# from pydantic import BaseModel
# from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Depends
# from sqlalchemy.orm import Session
# from sqlalchemy import text
# from typing import Optional

# from fastapi import Query
# import numpy as np
# import cv2
# import os
# import re
# import io
# import uuid
# import base64
# import json
# import time
# from PIL import Image
# from threading import Thread
# from datetime import datetime

# from config.db import get_db
# from preprocessing import prepare_image_for_ocr  # step by step langsung
# from ocr_engine import extract_text              # bukan lewat pipeline lagi
# from rbs import extract_metadata



# # ================================================================
# # VALIDASI FILE
# # ================================================================
# _MAX_FILE_SIZE   = 3 * 1024 * 1024                        # 3 MB
# _ALLOWED_EXT     = {".pdf", ".jpg", ".jpeg", ".png"}
# _ALLOWED_EXT_STR = "PDF, JPG, JPEG, PNG"

# router = APIRouter()



# # ================================================================

# # IN-MEMORY JOB STORE

# # Menyimpan status & hasil setiap job OCR yang sedang/sudah berjalan.

# # Format: { job_id: { "status": "...", "result": {...}, "created_at": float } }

# #

# # Status lifecycle:

# #   queued → preprocessing → ocr → generating_pdf → done

# #                                                  → error

# # ================================================================

# _jobs: dict = {}



# def _cleanup_old_jobs():

#     """Hapus job yang sudah lebih dari 1 jam agar tidak bocor memori."""

#     now = time.time()

#     expired = [jid for jid, data in _jobs.items() if now - data.get("created_at", now) > 3600]

#     for jid in expired:

#         del _jobs[jid]





# # ================================================================

# # BACKGROUND WORKER

# # Dipanggil di thread terpisah. Update status di setiap step

# # supaya Flutter bisa polling dan progress bar jalan real.

# # ================================================================

# def _run_ocr_job(job_id: str, image, coords_json, screen_w, screen_h, jenis_surat: str = "masuk"):

#     try:

#         # --- STEP 1: Parse koordinat manual (kalau ada) ---

#         manual_pts = None

#         if coords_json and screen_w and screen_h:

#             try:

#                 pts = np.array(json.loads(coords_json), dtype="float32")

#                 img_h, img_w = image.shape[:2]

#                 ratio_w = img_w / float(screen_w)

#                 ratio_h = img_h / float(screen_h)

#                 manual_pts = pts * [ratio_w, ratio_h]

#                 print(f"✔ [JOB {job_id}] Pakai manual points | ratio: {ratio_w:.2f}, {ratio_h:.2f}")

#             except Exception as e:

#                 print(f"⚠ [JOB {job_id}] Gagal parsing koordinat: {e}")



#         # --- STEP 2: Preprocessing (crop + enhance) ---

#         _jobs[job_id]["status"] = "preprocessing"

#         print(f"▶ [JOB {job_id}] Step 1: Preprocessing...")

#         img_color, img_ocr = prepare_image_for_ocr(image, manual_points=manual_pts)



#         # --- STEP 3: OCR ---

#         _jobs[job_id]["status"] = "ocr"

#         print(f"▶ [JOB {job_id}] Step 2: Extracting text...")

#         raw_text = extract_text(img_ocr)



#         # --- STEP 4: Generate PDF ---

#         _jobs[job_id]["status"] = "generating_pdf"

#         print(f"▶ [JOB {job_id}] Step 3: Generating PDF...")

#         img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

#         pil_img = Image.fromarray(img_rgb)

#         pdf_io = io.BytesIO()

#         pil_img.save(pdf_io, format="PDF", resolution=100.0)

#         pdf_base64 = base64.b64encode(pdf_io.getvalue()).decode()



#         # --- STEP 5: Extract metadata ---

#         metadata = extract_metadata(raw_text, jenis_surat=jenis_surat)



#         # --- DONE ---

#         _jobs[job_id]["status"] = "done"

#         _jobs[job_id]["result"] = {

#             "success": True,

#             "text": raw_text,

#             "metadata": metadata,

#             "pdf_base64": pdf_base64,

#         }

#         print(f"✅ [JOB {job_id}] Selesai.")



#     except Exception as e:

#         print(f"❌ [JOB {job_id}] Error: {str(e)}")

#         _jobs[job_id]["status"] = "error"

#         _jobs[job_id]["message"] = str(e)





# # ================================================================

# # POST /api/ocr

# # Terima file → simpan job → jalankan worker di thread → return job_id

# # Response langsung (<5 detik), tidak ada timeout lagi.

# # ================================================================


# @router.post("/api/ocr")
# async def ocr_endpoint(
#     file: UploadFile = File(...),
#     coords: Optional[str] = Form(None),
#     width: Optional[float] = Form(None),
#     height: Optional[float] = Form(None),
#     jenis_surat: Optional[str] = Form("masuk"),
# ):
#     try:
#         content = await file.read()
#         filename = file.filename.lower()

#         # ── VALIDASI TIPE & UKURAN ──────────────────────────────
#         ext = os.path.splitext(filename)[1]
#         if ext not in _ALLOWED_EXT:
#             raise HTTPException(400, f"Tipe file tidak didukung. Gunakan: {_ALLOWED_EXT_STR}.")
#         if len(content) > _MAX_FILE_SIZE:
#             raise HTTPException(400, f"Ukuran file terlalu besar. Maksimal {_MAX_FILE_SIZE // (1024*1024)}MB.")
#         # ────────────────────────────────────────────────────────




#         # ── PDF → konversi halaman pertama ke image dulu ──

#         if filename.endswith(".pdf"):

#             try:

#                 import fitz  # PyMuPDF
#                 pdf_doc = fitz.open(stream=content, filetype="pdf")
#                 page = pdf_doc[0]  # ambil halaman pertama
#                 mat = fitz.Matrix(2.0, 2.0)  # scale 2x biar resolusi bagus
#                 pix = page.get_pixmap(matrix=mat)
#                 img_bytes = pix.tobytes("png")
#                 nparr = np.frombuffer(img_bytes, np.uint8)
#                 image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#                 pdf_doc.close()

#             except Exception as e:

#                 raise HTTPException(400, f"Gagal membaca PDF: {str(e)}")



#         # ── JPG/PNG → langsung decode ──

#         else:

#             nparr = np.frombuffer(content, np.uint8)

#             image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)



#         if image is None:

#             raise HTTPException(400, "File tidak dapat diproses.")



#         # Buat job baru

#         job_id = str(uuid.uuid4())

#         _jobs[job_id] = {

#             "status": "queued",

#             "result": None,

#             "message": None,

#             "created_at": time.time(),

#         }



#         thread = Thread(

#             target=_run_ocr_job,

#             args=(job_id, image, coords, width, height, jenis_surat),

#             daemon=True

#         )

#         thread.start()



#         _cleanup_old_jobs()



#         print(f"🚀 [JOB {job_id}] Diterima & antri diproses.")

#         return {"success": True, "job_id": job_id}



#     except HTTPException:

#         raise

#     except Exception as e:

#         print(f"ERROR OCR ENDPOINT: {str(e)}")

#         raise HTTPException(500, f"Terjadi kesalahan: {str(e)}")



# # ================================================================

# # GET /api/ocr/status/{job_id}

# # Dipanggil Flutter tiap 2 detik untuk cek progress.

# # Kalau status "done" → ikutkan result sekalian.

# # Kalau status "error" → ikutkan pesan error.

# # ================================================================

# @router.get("/api/ocr/status/{job_id}")

# async def ocr_status(job_id: str):

#     job = _jobs.get(job_id)



#     if job is None:

#         raise HTTPException(404, f"Job '{job_id}' tidak ditemukan atau sudah kedaluwarsa.")



#     status = job["status"]



#     if status == "done":

#         return {

#             "status": "done",

#             **job["result"]   # spread: success, text, metadata, pdf_base64

#         }



#     if status == "error":

#         return {

#             "status": "error",

#             "message": job.get("message", "Terjadi kesalahan tidak diketahui.")

#         }



#     # Masih berjalan → return status saja (queued / preprocessing / ocr / generating_pdf)

#     return {"status": status}





# # ================================================================

# # POST /api/simpan-arsip  (tidak berubah)

# # ================================================================

# @router.post("/api/simpan-arsip")

# async def simpan_arsip(

#     file: UploadFile = File(...),
#     nomor_surat: str = Form(...),
#     tanggal_surat: str = Form(...),
#     perihal: str = Form(...),
#     jenis_surat: str = Form(...),
#     asal_tujuan: str = Form(...),
#     id_user_input: int = Form(...),
#     id_unit: int = Form(...),
#     db: Session = Depends(get_db)

# ):

#     try:

#         pdf_bytes = await file.read()


#         # ── VALIDASI TIPE & UKURAN ──────────────────────────────
#         ext = os.path.splitext(file.filename.lower())[1]
#         if ext not in _ALLOWED_EXT:
#             raise HTTPException(400, f"Tipe file tidak didukung. Gunakan: {_ALLOWED_EXT_STR}.")
#         if len(pdf_bytes) > _MAX_FILE_SIZE:
#             raise HTTPException(400, f"Ukuran file terlalu besar. Maksimal 2MB.")
#         # ────────────────────────────────────────────────────────


#         # ══════════════════════════════════════════
#         # VALIDASI DUPLIKAT NOMOR SURAT
#         # Cek apakah nomor_surat + jenis_surat sudah ada di DB
#         # ══════════════════════════════════════════
#         cek_duplikat = db.execute(
#             text("""
#                 SELECT nomor_surat, tanggal_surat, perihal
#                 FROM surat
#                 WHERE nomor_surat = :nomor
#                 AND jenis_surat   = :jenis
#                 LIMIT 1
#             """),
#             {"nomor": nomor_surat.strip(), "jenis": jenis_surat}
#         ).fetchone()

#         if cek_duplikat:
#             return {
#                 "success"  : False,
#                 "duplikat" : True,
#                 "message"  : f"Nomor surat '{nomor_surat}' sudah tercatat di arsip {jenis_surat}.",
#                 "data_lama": {
#                     "nomor_surat"  : cek_duplikat.nomor_surat,
#                     "tanggal_surat": str(cek_duplikat.tanggal_surat),
#                     "perihal"      : cek_duplikat.perihal,
#                 }
#             }
#         # ══════════════════════════════════════════


#         col_count = "asal_surat" if jenis_surat == "masuk" else "tujuan_surat"
#         count_query = text(f"SELECT COUNT(*) FROM surat WHERE {col_count} = :val")
#         result_count = db.execute(count_query, {"val": asal_tujuan}).scalar()



#         prefix = "m" if jenis_surat == "masuk" else "k"
#         urutan = (result_count or 0) + 1

#         try:
#             tgl_obj = datetime.strptime(tanggal_surat, "%Y-%m-%d")
#         except Exception:
#             tgl_obj = datetime.now()


#         safe_asal_tujuan = re.sub(r'[^a-zA-Z0-9]', '', asal_tujuan.lower())
#         filename = f"{tgl_obj.strftime('%Y%m%d')}_{safe_asal_tujuan}_{prefix}{urutan}.pdf"

#         folder_path = "uploads/arsip"
#         if not os.path.exists(folder_path):
#             os.makedirs(folder_path)

#         file_path_fisik = os.path.join(folder_path, filename)

#         with open(file_path_fisik, "wb") as f:
#             f.write(pdf_bytes)

#         db_path = file_path_fisik.replace("\\", "/")

#         sql = text(f"""
#             INSERT INTO surat (
#                 nomor_surat,
#                 tanggal_surat,
#                 perihal,
#                 jenis_surat,
#                 {col_count},
#                 file_path,
#                 id_user_input,
#                 id_unit,
#                 created_at
#             ) VALUES (:no, :tgl, :hal, :jenis, :asal, :path, :uid, :unit, NOW())
#         """)

#         db.execute(sql, {
#             "no": nomor_surat,
#             "tgl": tgl_obj.date(),
#             "hal": perihal,
#             "jenis": jenis_surat,
#             "asal": asal_tujuan,
#             "path": db_path,
#             "uid": id_user_input,
#             "unit": id_unit
#         })

#         db.commit()

#         return {
#             "success": True,
#             "message": "Arsip Universitas Prabumulih Berhasil Disimpan!",
#             "filename": filename
#         }

#     except Exception as e:
#         db.rollback()
#         print(f"DETAIL ERROR DB: {str(e)}")
#         return {"success": False, "message": f"Gagal masuk Database: {str(e)}"}



# # ================================================================

# # GET /api/surat-masuk

# # ================================================================

# @router.get("/api/surat-masuk")

# async def get_surat_masuk(

#     role: str = Query(None), 

#     id_unit: int = Query(None), 

#     db: Session = Depends(get_db)

# ):

#     try:

#         # Base query

#         sql_string = """

#             SELECT

#                 s.*,

#                 COUNT(d.id_disposisi) as jml_disposisi

#             FROM surat s

#             LEFT JOIN disposisi d ON s.id_surat = d.id_surat

#             WHERE s.jenis_surat = 'masuk'

#         """

#         params = {}



#         # Logika Hak Akses: Kalau dia operator, filter berdasarkan id_unit

#         if role and role.lower().strip() == "operator" and id_unit:

#             sql_string += " AND s.id_unit = :id_unit"

#             params["id_unit"] = id_unit



#         sql_string += " GROUP BY s.id_surat ORDER BY s.created_at DESC"



#         sql = text(sql_string)

#         result = db.execute(sql, params).mappings().all()



#         final_data = []

#         for row in result:

#             new_row = dict(row)

#             new_row['status_label'] = "Lihat Disposisi" if new_row['jml_disposisi'] > 0 else "Belum Disposisi"

#             final_data.append(new_row)



#         return {"success": True, "data": final_data}

#     except Exception as e:

#         return {"success": False, "message": str(e)}





# # ================================================================

# # GET /api/surat-keluar

# # ================================================================

# @router.get("/api/surat-keluar")

# async def get_surat_keluar(

#     role: str = Query(None), 

#     id_unit: int = Query(None), 

#     db: Session = Depends(get_db)

# ):

#     try:

#         sql_string = """

#             SELECT s.*

#             FROM surat s

#             WHERE s.jenis_surat = 'keluar'

#         """

#         params = {}



#         # Logika Hak Akses yang sama untuk surat keluar

#         if role and role.lower().strip() == "operator" and id_unit:

#             sql_string += " AND s.id_unit = :id_unit"

#             params["id_unit"] = id_unit



#         sql_string += " ORDER BY s.created_at DESC"



#         sql = text(sql_string)

#         result = db.execute(sql, params).mappings().all()



#         return {"success": True, "data": [dict(row) for row in result]}



#     except Exception as e:

#         return {"success": False, "message": str(e)}





# # ================================================================

# # POST /api/disposisi  (tidak berubah)

# # ================================================================

# class DisposisiRequest(BaseModel):

#     id_surat: int

#     id_unit: int

#     catatan: str

#     id_user_pimpinan: int

#     sifat_surat: str



# @router.post("/api/disposisi")

# async def buat_disposisi(data: DisposisiRequest, db: Session = Depends(get_db)):

#     try:

#         sql = text("""

#             INSERT INTO disposisi (id_surat, id_unit, isi_disposisi, id_user, tanggal_disposisi, sifat_surat)

#             VALUES (:id_surat, :id_unit, :catatan, :id_user, :tgl, :sifat_surat)

#         """)



#         db.execute(sql, {

#             "id_surat": data.id_surat,

#             "id_unit": data.id_unit,

#             "catatan": data.catatan,

#             "id_user": data.id_user_pimpinan,

#             "tgl": datetime.now(),

#             "sifat_surat": data.sifat_surat 

#         })

#         db.commit()



#         print("--- DISPOSISI BERHASIL DISIMPAN ---")

#         return {"success": True, "message": "Disposisi berhasil disimpan"}



#     except Exception as e:

#         db.rollback()

#         print(f"ERROR SIMPAN DISPOSISI: {str(e)}")

#         raise HTTPException(status_code=500, detail=str(e))











# # ================================================================

# # GET /api/disposisi/{id_surat}

# # Ambil semua disposisi berdasarkan id_surat

# # ================================================================

# @router.get("/api/disposisi/{id_surat}")

# async def get_disposisi(id_surat: int, db: Session = Depends(get_db)):

#     try:

#         sql = text("""

#             SELECT

#                 d.id_disposisi,

#                 d.id_surat,

#                 d.isi_disposisi,

#                 d.sifat_surat,

#                 DATE_FORMAT(d.tanggal_disposisi, '%d-%m-%Y') AS tanggal_disposisi,

#                 un.nama_unit,

#                 s.nomor_surat,

#                 s.tanggal_surat,

#                 s.asal_surat,

#                 s.perihal

#             FROM disposisi d

#             LEFT JOIN unit  un ON d.id_unit  = un.id_unit

#             LEFT JOIN surat s  ON d.id_surat = s.id_surat

#             WHERE d.id_surat = :id_surat

#             ORDER BY d.tanggal_disposisi DESC

#         """)





#         result = db.execute(sql, {"id_surat": id_surat}).mappings().all()

#         print(f"DEBUG disposisi id_surat={id_surat} → {len(result)} rows")



#         return {"success": True, "data": [dict(row) for row in result]}



#     except Exception as e:

#         print(f"ERROR GET DISPOSISI: {str(e)}")

#         return {"success": False, "message": str(e)}









# # ================================================================

# # DELETE /api/disposisi/{id_disposisi}

# # Hapus satu instruksi disposisi berdasarkan id_disposisi

# # ================================================================

# @router.delete("/api/disposisi/{id_disposisi}")

# async def hapus_disposisi(id_disposisi: int, db: Session = Depends(get_db)):

#     try:

#         # Cek dulu apakah data ada

#         cek = db.execute(

#             text("SELECT id_disposisi FROM disposisi WHERE id_disposisi = :id"),

#             {"id": id_disposisi}

#         ).fetchone()



#         if not cek:

#             raise HTTPException(status_code=404, detail="Disposisi tidak ditemukan")



#         db.execute(

#             text("DELETE FROM disposisi WHERE id_disposisi = :id"),

#             {"id": id_disposisi}

#         )

#         db.commit()



#         print(f"✅ Disposisi id={id_disposisi} berhasil dihapus")

#         return {"success": True, "message": "Instruksi disposisi berhasil dihapus"}



#     except HTTPException:

#         raise

#     except Exception as e:

#         db.rollback()

#         print(f"ERROR HAPUS DISPOSISI: {str(e)}")

#         raise HTTPException(status_code=500, detail=str(e))













# # DELETE /api/surat/{id_surat}

# @router.delete("/api/surat/{id_surat}")

# async def hapus_surat(id_surat: int, db: Session = Depends(get_db)):

#     try:

#         cek = db.execute(

#             text("SELECT id_surat, file_path FROM surat WHERE id_surat = :id"),

#             {"id": id_surat}

#         ).fetchone()



#         if not cek:

#             raise HTTPException(status_code=404, detail="Surat tidak ditemukan")



#         # Hapus file fisik juga

#         if cek.file_path and os.path.exists(cek.file_path):

#             os.remove(cek.file_path)



#         db.execute(text("DELETE FROM surat WHERE id_surat = :id"), {"id": id_surat})

#         db.commit()



#         return {"success": True, "message": "Arsip berhasil dihapus"}



#     except HTTPException:

#         raise

#     except Exception as e:

#         db.rollback()

#         raise HTTPException(status_code=500, detail=str(e))









# # ================================================================

# # PUT /api/surat/{id_surat}

# # Edit metadata arsip — bisa masuk & keluar

# # ================================================================

# @router.put("/api/surat/{id_surat}")

# async def edit_surat(id_surat: int, data: dict, db: Session = Depends(get_db)):

#     try:

#         # Cek surat ada tidak + ambil jenis_suratnya

#         cek = db.execute(

#             text("SELECT id_surat, jenis_surat FROM surat WHERE id_surat = :id"),

#             {"id": id_surat}

#         ).fetchone()



#         if not cek:

#             raise HTTPException(status_code=404, detail="Surat tidak ditemukan")



#         # Tentukan kolom asal/tujuan otomatis berdasarkan jenis_surat di DB

#         col_asal_tujuan = "asal_surat" if cek.jenis_surat == "masuk" else "tujuan_surat"

#         val_asal_tujuan = data.get("asal_surat") or data.get("tujuan_surat", "")



#         db.execute(

#             text(f"""

#                 UPDATE surat SET

#                     nomor_surat       = :no,

#                     tanggal_surat     = :tgl,

#                     perihal           = :hal,

#                     {col_asal_tujuan} = :asal_tujuan

#                 WHERE id_surat = :id

#             """),

#             {

#                 "no":          data.get("nomor_surat", ""),

#                 "tgl":         data.get("tanggal_surat", ""),

#                 "hal":         data.get("perihal", ""),

#                 "asal_tujuan": val_asal_tujuan,

#                 "id":          id_surat,

#             }

#         )

#         db.commit()



#         return {"success": True, "message": "Arsip berhasil diperbarui"}



#     except HTTPException:

#         raise

#     except Exception as e:

#         db.rollback()

#         print(f"ERROR EDIT SURAT: {str(e)}")

#         raise HTTPException(status_code=500, detail=str(e))





# # ================================================================

# # POST /api/simpan-surat-keluar

# # Simpan metadata surat keluar dari editor (JSON, tanpa file).

# # Dipanggil oleh EditorSuratKeluarPage di Flutter.

# # ================================================================

# class SuratKeluarRequest(BaseModel):

#     nomor_surat: str

#     tanggal_surat: str

#     perihal: str

#     tujuan_surat: str

#     id_user_input: int

#     id_unit: int



# @router.post("/api/simpan-surat-keluar")

# async def simpan_surat_keluar(data: SuratKeluarRequest, db: Session = Depends(get_db)):

#     try:

#         try:

#             tgl_obj = datetime.strptime(data.tanggal_surat, "%Y-%m-%d")

#         except Exception:

#             tgl_obj = datetime.now()



#         sql = text("""

#             INSERT INTO surat (

#                 nomor_surat,

#                 tanggal_surat,

#                 perihal,

#                 jenis_surat,

#                 tujuan_surat,

#                 id_user_input,

#                 id_unit,

#                 created_at

#             ) VALUES (:no, :tgl, :hal, 'keluar', :tujuan, :uid, :unit, NOW())

#         """)



#         db.execute(sql, {

#             "no"    : data.nomor_surat,

#             "tgl"   : tgl_obj.date(),

#             "hal"   : data.perihal,

#             "tujuan": data.tujuan_surat,

#             "uid"   : data.id_user_input,

#             "unit"  : data.id_unit,

#         })

#         db.commit()



#         print(f"✅ Surat keluar '{data.nomor_surat}' berhasil disimpan.")

#         return {"success": True, "message": "Metadata surat keluar berhasil disimpan!"}



#     except Exception as e:

#         db.rollback()

#         print(f"ERROR SIMPAN SURAT KELUAR: {str(e)}")

#         return {"success": False, "message": f"Gagal menyimpan: {str(e)}"}



# @router.put("/api/surat/{id_surat}/upload-ttd")
# async def upload_ttd(
#     id_surat : int,
#     file     : UploadFile = File(...),
#     db       : Session    = Depends(get_db)
# ):
#     try:
#         cek = db.execute(
#             text("SELECT id_surat, file_path FROM surat WHERE id_surat = :id"),
#             {"id": id_surat}
#         ).fetchone()
 
#         if not cek:
#             raise HTTPException(status_code=404, detail="Surat tidak ditemukan.")
 
#         # Hapus file lama kalau sudah ada sebelumnya
#         if cek.file_path and os.path.exists(cek.file_path):
#             os.remove(cek.file_path)
 
#         # Validasi ekstensi
#         ext = os.path.splitext(file.filename)[1].lower()
#         if ext not in {".pdf", ".jpg", ".jpeg", ".png"}:
#             raise HTTPException(status_code=400, detail="Format file tidak didukung. Gunakan PDF, JPG, atau PNG.")
 
#         # Simpan file baru dengan nama unik
#         folder = "uploads/arsip"
#         os.makedirs(folder, exist_ok=True)
#         new_filename = f"{uuid.uuid4().hex}_ttd{ext}"
#         new_path     = os.path.join(folder, new_filename)
 
#         raw_bytes = await file.read()
#         with open(new_path, "wb") as f:
#             f.write(raw_bytes)
 
#         db_path = new_path.replace("\\", "/")
 
#         # Update file_path di database
#         db.execute(
#             text("UPDATE surat SET file_path = :path WHERE id_surat = :id"),
#             {"path": db_path, "id": id_surat}
#         )
#         db.commit()
 
#         return {
#             "success"  : True,
#             "message"  : "PDF TTD berhasil diupload.",
#             "file_path": db_path
#         }
 
#     except HTTPException:
#         raise
#     except Exception as e:
#         db.rollback()
#         print(f"ERROR upload-ttd: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))
 
 





# # ================================================================
# # GET /api/surat-keluar
# # ================================================================
# # @router.get("/api/surat-keluar")
# # async def get_surat_keluar(db: Session = Depends(get_db)):
# #     try:
# #         sql = text("""
# #             SELECT s.*
# #             FROM surat s
# #             WHERE s.jenis_surat = 'keluar'
# #             ORDER BY s.created_at DESC
# #         """)

# #         result = db.execute(sql).mappings().all()

# #         return {"success": True, "data": [dict(row) for row in result]}

# #     except Exception as e:
# #         return {"success": False, "message": str(e)}


        



# # import os
# # import uuid
# # import pymysql
# # import logging
# # import base64

# # from fastapi import APIRouter, File, UploadFile, Form, HTTPException
# # from typing import Optional

# # # Import pendukung (Pastikan ocr.py dan rbs.py ada di folder yang sama)
# # from ocr import process_full_ocr 
# # from ocr import process_manual_crop
# # from rbs import extract_metadata as process_rules

# # router = APIRouter()

# # # ==========================================
# # # KONFIGURASI (Universitas Prabumulih)
# # # ==========================================
# # UPLOAD_DIR = "uploads/arsip"
# # MAX_FILE_SIZE = 10 * 1024 * 1024 
# # ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png", "pdf"]

# # os.makedirs(UPLOAD_DIR, exist_ok=True)
# # logging.basicConfig(level=logging.INFO)

# # def get_db_connection():
# #     return pymysql.connect(
# #         host='localhost',
# #         user='root',
# #         password='@Ciciwulan24', 
# #         database='arsip_digital',
# #         cursorclass=pymysql.cursors.DictCursor
# #     )

# # # ==========================================
# # # HELPER: Validasi & Nama File
# # # ==========================================
# # def validate_file(filename: str, size: int):
# #     ext = filename.split('.')[-1].lower()
# #     if ext not in ALLOWED_EXTENSIONS:
# #         raise HTTPException(400, f"Format {ext} tidak didukung!")
# #     if size > MAX_FILE_SIZE:
# #         raise HTTPException(400, "Ukuran file terlalu besar (Max 10MB)")
# #     return ext

# # def generate_safe_filename(nomor_surat: str, ext: str):
# #     # Membersihkan karakter ilegal agar tidak error di OS Windows
# #     safe_no = "".join([c if c.isalnum() else "_" for c in nomor_surat])
# #     unique_id = uuid.uuid4().hex[:6]
# #     return f"arsip_{safe_no}_{unique_id}.{ext}"


# # # ==========================================
# # # ENDPOINT 1: OCR + RBS (EXTRACTION)
# # # ==========================================
# # @router.post("/api/ocr-extract")
# # async def extract_arsip(
# #     file: UploadFile = File(...),
# #     id_user: int = Form(...),
# #     jenis_surat: str = Form(...) 
# # ):
# #     try:
# #         content = await file.read()
# #         validate_file(file.filename, len(content))

# #         # 1. Jalankan OCR
# #         # process_full_ocr di ocr.py harus mengembalikan:
# #         # raw_text: Teks hasil OCR
# #         # warped_base64: Gambar yang sudah diluruskan (dalam format base64 untuk preview di Flutter)
# #         # _ : (bytes pdf jika ada)
# #         raw_text, warped_base64, _ = process_full_ocr(content, file.filename)

# #         if not raw_text.strip():
# #             raise HTTPException(400, "Teks tidak terdeteksi. Pastikan dokumen tegak lurus ya Ci.")

# #         # 2. Jalankan RBS 
# #         metadata = process_rules(raw_text, jenis_surat)

# #         return {
# #             "success": True,
# #             "data": {
# #                 "asal_tujuan": metadata["asal_surat"],
# #                 "nomor_surat": metadata["nomor_surat"],
# #                 "tanggal_surat": metadata["tanggal_surat"],
# #                 "perihal": metadata["perihal"]
# #             },
# #             "raw_text": raw_text,
# #             "preview_image": warped_base64 # Ini gambar yang sudah rapi/lurus
# #         }

# #     except Exception as e:
# #         logging.error(f"EXTRACTION ERROR: {e}")
# #         raise HTTPException(500, f"Terjadi kesalahan saat ekstraksi: {str(e)}")






# # # ==========================================
# # # ENDPOINT 2: SIMPAN KE DATABASE
# # # ==========================================
# # @router.post("/api/simpan-arsip")
# # async def simpan_ke_db(
# #     file: UploadFile = File(...),
# #     id_surat: Optional[str] = Form(None),
# #     id_user: int = Form(...),
# #     jenis_surat: str = Form(...),
# #     nomor_surat: str = Form(...),
# #     tanggal_surat: str = Form(...),
# #     perihal: str = Form(...),
# #     asal_tujuan: str = Form(...),
# #     id_user_input: int = Form(...),
# #     id_unit: int = Form(...)
# # ):
# #     connection = None
# #     try:
# #         content = await file.read()
        
# #         # Panggil kembali fungsi OCR untuk mendapatkan 'clean_pdf_bytes'
# #         # Di dalam ocr.py, pastikan fungsi ini memproses image -> warping -> search-able PDF
# #         _, _, clean_pdf_bytes = process_full_ocr(content, file.filename)
        
# #         # Logika pemilihan file: Prioritaskan PDF hasil olahan mesin OCR kita
# #         if clean_pdf_bytes:
# #             save_content = clean_pdf_bytes
# #             save_ext = "pdf"
# #         else:
# #             save_content = content
# #             save_ext = file.filename.split('.')[-1].lower()

# #         filename = generate_safe_filename(nomor_surat, save_ext)
# #         file_path = os.path.join(UPLOAD_DIR, filename)

# #         # Simpan file fisik ke folder uploads/arsip
# #         with open(file_path, "wb") as f:
# #             f.write(save_content)

# #         # ... lanjut ke proses insert database (sama seperti kode Cici sebelumnya) ...

# #         db_path = file_path.replace("\\", "/") 

# #         connection = get_db_connection()
# #         with connection.cursor() as cursor:
# #             # SINKRONISASI DATABASE:
# #             # Jika 'masuk', masuk kolom asal_surat. Jika 'keluar', masuk kolom tujuan_surat.
# #             column_target = "asal_surat" if jenis_surat == 'masuk' else "tujuan_surat"

# #             sql = f"""
# #             INSERT INTO surat 
# #             (id_surat, nomor_surat, tanggal_surat, perihal, jenis_surat, 
# #              {column_target}, id_user_input, id_unit, file_path)
# #             VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
# #             """

# #             cursor.execute(sql, (
# #                 str(uuid.uuid4()), nomor_surat, tanggal_surat, perihal, 
# #                 jenis_surat, asal_tujuan, id_user_input, id_unit, db_path
# #             ))

# #             connection.commit()

# #         return {"success": True, "message": "Arsip Universitas Prabumulih Berhasil Disimpan!"}

# #     except Exception as e:
# #         logging.error(f"SAVE ERROR: {e}")
# #         raise HTTPException(500, f"Gagal menyimpan ke database: {str(e)}")
# #     finally:
# #         if connection:
# #             connection.close()

# # # ==========================================
# # # ENDPOINT 3: DAFTAR SURAT (Untuk Tabel UI)
# # # ==========================================
# # @router.get("/api/surat-masuk")
# # async def get_surat_masuk():
# #     connection = None
# #     try:
# #         connection = get_db_connection()
# #         with connection.cursor() as cursor:
# #             cursor.execute("""
# #                 SELECT * FROM surat 
# #                 WHERE jenis_surat = 'masuk'
# #                 ORDER BY created_at DESC
# #             """)
# #             result = cursor.fetchall()
# #         return {"success": True, "data": result}
# #     except Exception as e:
# #         logging.error(f"FETCH ERROR: {e}")
# #         raise HTTPException(500, "Gagal memuat daftar surat masuk")
# #     finally:
# #         if connection:
# #             connection.close()