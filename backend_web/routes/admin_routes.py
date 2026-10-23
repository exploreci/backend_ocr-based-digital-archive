from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text  # Penting untuk menjalankan query raw
from config.db import get_db
from datetime import datetime

router = APIRouter()

# 1. AMBIL SEMUA UNIT
@router.get("/api/unit")
async def get_all_units(db: Session = Depends(get_db)):
    try:
        # Gunakan db.execute(text(...)) karena db adalah Session SQLAlchemy
        sql = text("SELECT id_unit, nama_unit, deskripsi, is_activate, created_at FROM unit ORDER BY id_unit ASC")
        result = db.execute(sql).mappings().all() # .mappings() agar hasilnya jadi dictionary
        return {"success": True, "data": list(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database Error: {str(e)}")

# 2. TAMBAH UNIT BARU
@router.post("/api/unit")
async def add_unit(data: dict, db: Session = Depends(get_db)):
    try:
        sql = text("""
            INSERT INTO unit (nama_unit, deskripsi, is_activate, created_at) 
            VALUES (:nama, :desc, :active, :created)
        """)
        db.execute(sql, {
            "nama": data['nama_unit'],
            "desc": data.get('deskripsi', ''),
            "active": data.get('is_activate', 1),
            "created": datetime.now()
        })
        db.commit()
        return {"success": True, "message": "Unit berhasil ditambahkan"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

# 3. UPDATE UNIT
@router.put("/api/unit/{id_unit}")
async def update_unit(id_unit: int, data: dict, db: Session = Depends(get_db)):
    try:
        sql = text("""
            UPDATE unit 
            SET nama_unit = :nama, deskripsi = :desc, is_activate = :active 
            WHERE id_unit = :id
        """)
        db.execute(sql, {
            "nama": data['nama_unit'],
            "desc": data['deskripsi'],
            "active": data['is_activate'],
            "id": id_unit
        })
        db.commit()
        return {"success": True, "message": "Unit berhasil diperbarui"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

# 4. HAPUS UNIT
@router.delete("/api/unit/{id_unit}")
async def delete_unit(id_unit: int, db: Session = Depends(get_db)):
    try:
        sql = text("DELETE FROM unit WHERE id_unit = :id")
        db.execute(sql, {"id": id_unit})
        db.commit()
        return {"success": True, "message": "Unit berhasil dihapus"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))