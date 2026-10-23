from fastapi import Request, HTTPException, Depends
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from config.db import get_db
from utils.auth import hash_password, check_password, generate_token
from sqlalchemy import text
import smtplib
import secrets
import re
from datetime import datetime, timedelta
from email.message import EmailMessage
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


def validate_password(password: str):
    """
    Validasi aturan keamanan password:
    - Minimal 8 karakter
    - Mengandung huruf besar (A-Z)
    - Mengandung huruf kecil (a-z)
    - Mengandung angka (0-9)
    - Mengandung simbol (!@#$%^&* dst)
    """
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password minimal 8 karakter!")
    if not re.search(r'[A-Z]', password):
        raise HTTPException(status_code=400, detail="Password harus mengandung minimal 1 huruf besar!")
    if not re.search(r'[a-z]', password):
        raise HTTPException(status_code=400, detail="Password harus mengandung minimal 1 huruf kecil!")
    if not re.search(r'[0-9]', password):
        raise HTTPException(status_code=400, detail="Password harus mengandung minimal 1 angka!")
    if not re.search(r'[^A-Za-z0-9]', password):
        raise HTTPException(status_code=400, detail="Password harus mengandung minimal 1 simbol (contoh: @, #, !, _)!")



router = APIRouter(prefix="/auth", tags=["Auth"])


# 🔐 REGISTER
from fastapi import HTTPException

@router.post("/register")
def register(data: dict, db: Session = Depends(get_db)):
    # 1. Cek apakah username atau email sudah ada
    check_query = text("SELECT id_user FROM users WHERE username = :u OR email = :e")
    existing_user = db.execute(check_query, {"u": data['username'], "e": data['email']}).fetchone()

    if existing_user:
        raise HTTPException(status_code=400, detail="Username atau Email sudah terdaftar")

    # 2. Validasi kekuatan password
    validate_password(data['password'])

    # 3. Jika aman, baru lakukan Insert
    hashed = hash_password(data['password'])
    query = text("""
    INSERT INTO users (username, nama_lengkap, email, password, id_role, id_unit, id_unit_akses, created_at)
    VALUES (:username, :nama_lengkap, :email, :password, :id_role, :id_unit, :id_unit_akses, NOW())
    """)

    db.execute(query, {
        "username": data['username'],
        "nama_lengkap": data['nama_lengkap'],
        "email": data['email'],
        "password": hashed,
        "id_role": data['id_role'],
        "id_unit": data['id_unit'],
        "id_unit_akses": data.get('id_unit_akses') # Pakai .get agar kalau kosong tidak error
    })
    db.commit()
    return {"success": True, "message": "User berhasil ditambahkan"}


@router.post("/login")
def login(data: dict, db: Session = Depends(get_db)):
    login_id = data.get("email") 
    password = data.get("password")

    print(f"\n--- DEBUG LOGIN ---")
    print(f"Mencoba login dengan ID: {login_id}")

    # 1. Tambahkan JOIN ke tabel unit agar nama_unit bisa diambil
    # 2. Gunakan :login_param agar konsisten
    query = text("""
        SELECT 
            u.id_user, u.username, u.nama_lengkap, u.email, u.password, 
            u.id_role, u.id_unit, u.id_unit_akses, 
            r.nama_role as role_name,
            n.nama_unit as unit_name
        FROM users u 
        JOIN role r ON u.id_role = r.id_role 
        JOIN unit n ON u.id_unit = n.id_unit
        WHERE u.email = :login_param OR u.username = :login_param
    """)

    # 3. Kirim parameter yang SAMA dengan di query (:login_param)
    result = db.execute(query, {"login_param": login_id})
    user = result.fetchone()

    if user:
        # Convert ke dict agar mudah diakses
        user_dict = dict(user._mapping)
        print(f"User ditemukan: {user_dict['username']} (Role: {user_dict['role_name']})")

        if check_password(password, user_dict['password']):
            print("Status: LOGIN BERHASIL.")
            token = generate_token(user_dict)

            return {
                "success": True,
                "token": token,
                "user": {
                    "id_user": user_dict["id_user"],
                    "nama": user_dict["nama_lengkap"],
                    "username": user_dict["username"],
                    "email": user_dict["email"],
                    "role": user_dict["role_name"], # Sesuaikan dengan alias di SQL
                    "id_role": user_dict["id_role"],
                    "unit": user_dict["unit_name"], # Sesuaikan dengan alias di SQL
                    "id_unit": user_dict["id_unit"],
                    "id_unit_akses": user_dict["id_unit_akses"] # Kirim ini juga Ci!
                }
            }
        else:
            print("Status: PASSWORD SALAH!")
            raise HTTPException(
                status_code=401,
                detail="Password salah"
            )
    else:
        print("Status: USER TIDAK DITEMUKAN!")
        raise HTTPException(
            status_code=404,
            detail="Akun tidak ditemukan"
        )




@router.get("/users")
def get_users(db: Session = Depends(get_db)):
    # Mengambil semua user beserta nama role dan nama unit-nya
    query = text("""
    SELECT 
        u.username, 
        u.nama_lengkap,
        u.email,
        u.id_role, 
        u.id_unit,
        u.id_unit_akses,
        r.nama_role as role, 
        un.nama_unit as unit
    FROM users u
    JOIN role r ON u.id_role = r.id_role
    JOIN unit un ON u.id_unit = un.id_unit
    ORDER BY u.id_user ASC
""")
    
    result = db.execute(query)
    users = [dict(row._mapping) for row in result.fetchall()]
    
    return {"success": True, "data": users}

@router.put("/users/{username}")
def update_user(username: str, data: dict, db: Session = Depends(get_db)):
    nama = data.get('nama_lengkap')
    email = data.get('email')
    role = data.get('id_role')
    unit = data.get('id_unit')
    password_baru = data.get('password')
    id_unit_akses = data.get('id_unit_akses')

    # ✅ FIX: Kalau id_unit_akses tidak dikirim atau null,
    # ambil nilai lama dari database supaya tidak tertimpa
    if id_unit_akses is None:
        existing = db.execute(
            text("SELECT id_unit_akses FROM users WHERE username = :u"),
            {"u": username}
        ).fetchone()
        if existing:
            id_unit_akses = existing.id_unit_akses

    sql = "UPDATE users SET nama_lengkap = :nama, email = :email, id_role = :role, id_unit = :unit, id_unit_akses = :id_unit_akses"
    params = {
        "nama": nama,
        "email": email,
        "role": role,
        "unit": unit,
        "id_unit_akses": id_unit_akses,
        "old_username": username
    }

    if password_baru and password_baru.strip() != "":
        sql += ", password = :password"
        params["password"] = hash_password(password_baru)

    sql += " WHERE username = :old_username"
    db.execute(text(sql), params)
    db.commit()

    return {"success": True, "message": "Data user berhasil diperbarui"}




# 🗑️ HAPUS USER
@router.delete("/users/{username}")
def delete_user(username: str, db: Session = Depends(get_db)):
    query = text("DELETE FROM users WHERE username = :u")
    db.execute(query, {"u": username})
    db.commit()
    return {"success": True, "message": "User berhasil dihapus"}


   # --- TAMBAHKAN INI DI auth_routes.py ---

@router.get("/roles")
def get_roles(db: Session = Depends(get_db)):
    # Mengambil daftar role untuk dropdown di Flutter
    result = db.execute(text("SELECT id_role, nama_role FROM role"))
    roles = [dict(row._mapping) for row in result.fetchall()]
    return {"success": True, "data": roles}

@router.get("/units")
def get_units(db: Session = Depends(get_db)):
    # Mengambil daftar unit untuk dropdown di Flutter
    result = db.execute(text("SELECT id_unit, nama_unit FROM unit"))
    units = [dict(row._mapping) for row in result.fetchall()]
    return {"success": True, "data": units} 






# Konfigurasi Email (Gunakan App Password Gmail)

EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

# --- LOGIKA RESET PASSWORD ---

def send_reset_email(target_email, token):
    msg = EmailMessage()
    msg['Subject'] = 'Reset Password Smart Web Arsip'
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = target_email
    
    # Link ini nantinya mengarah ke halaman reset di Flutter Web
    link = f"http://localhost:3000/#/reset-password?token={token}"
    
    msg.set_content(f"""
    Halo,
    
    Kami menerima permintaan untuk reset password akun Sistem Arsip Digital anda.
    Silakan klik link di bawah ini untuk membuat password baru:
    {link}
    
    Link ini hanya berlaku selama 15 menit. Jika anda tidak merasa meminta ini, abaikan saja.
    """)

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False



@router.post("/forgot-password")
async def forgot_password(request: dict, db: Session = Depends(get_db)):
    email = request.get("email")

    # Cek apakah email ada di DB
    check_user = db.execute(
        text("SELECT id_user FROM users WHERE email = :e"), 
        {"e": email}
    ).fetchone()
    
    if not check_user:
        raise HTTPException(status_code=404, detail="Email tidak terdaftar")

    # Generate token unik
    token = secrets.token_urlsafe(32)

    # ✅ Simpan ke tabel reset_password (nama baru)
    db.execute(
        text("""
            INSERT INTO reset_password 
            (email, token, is_used, created_at, updated_at)
            VALUES (:email, :token, FALSE, NOW(), NOW())
            ON DUPLICATE KEY UPDATE 
                token = :token, 
                updated_at = NOW()
        """),
        {"email": email, "token": token}
    )
    db.commit()

    success = send_reset_email(email, token)
    
    if success:
        return {"success": True, "message": "Link reset sudah dikirim ke email!"}
    else:
        raise HTTPException(status_code=500, detail="Gagal mengirim email")





@router.post("/reset-password-final")
async def reset_password_final(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    token = body.get("token")
    new_password = body.get("new_password")

    if not token or not new_password:
        raise HTTPException(status_code=400, detail="Data token atau password tidak lengkap!")

    # Validasi kekuatan password baru
    validate_password(new_password)

    try:
        hashed_pw = hash_password(new_password)
        
        # ✅ Cari email dari tabel reset_password (nama baru)
        reset_entry = db.execute(
            text("""
                SELECT email FROM reset_password 
                WHERE token = :t
                AND is_used = FALSE
                AND created_at >= NOW() - INTERVAL 15 MINUTE
            """),
            {"t": token}
        ).fetchone()

        if not reset_entry:
            raise HTTPException(status_code=400, detail="Token tidak valid atau kadaluwarsa")

        email_user = reset_entry[0]

        # Update password di tabel users
        db.execute(
            text("UPDATE users SET password = :p WHERE email = :e"),
            {"p": hashed_pw, "e": email_user}
        )
        
        # ✅ Hapus token dari tabel reset_password (nama baru)
        db.execute(
            text("""
                    UPDATE reset_password
                    SET is_used = TRUE,
                        updated_at = NOW()
                    WHERE token = :t
                """),
            {"t": token}
        )
        
        db.commit()
        return {"success": True, "message": "Password berhasil diperbarui"}

    except Exception as e:
        db.rollback()
        print(f"Error Reset: {e}")
        raise HTTPException(status_code=500, detail="Terjadi kesalahan pada server")



@router.post("/change-password")
async def change_password(request: Request, db: Session = Depends(get_db)):
    body = await request.json()
    
    id_user          = body.get("id_user")
    password_lama    = body.get("password_lama")
    password_baru    = body.get("password_baru")
    konfirmasi_password = body.get("konfirmasi_password")

    # 1. Validasi semua field terisi
    if not all([id_user, password_lama, password_baru, konfirmasi_password]):
        raise HTTPException(status_code=400, detail="Semua field wajib diisi!")

    # 2. Validasi password baru dan konfirmasi harus sama
    if password_baru != konfirmasi_password:
        raise HTTPException(status_code=400, detail="Password baru dan konfirmasi tidak cocok!")

    # 3. Validasi kekuatan password baru
    validate_password(password_baru)

    try:
        # 4. Ambil password lama dari DB berdasarkan id_user
        user = db.execute(
            text("SELECT password FROM users WHERE id_user = :id"),
            {"id": id_user}
        ).fetchone()

        if not user:
            raise HTTPException(status_code=404, detail="User tidak ditemukan!")

        # 5. Cek apakah password lama cocok
        if not check_password(password_lama, user[0]):
            raise HTTPException(status_code=400, detail="Password lama salah!")

        # 6. Pastikan password baru tidak sama dengan password lama
        if check_password(password_baru, user[0]):
            raise HTTPException(status_code=400, detail="Password baru tidak boleh sama dengan password lama!")

        # 7. Hash password baru lalu update ke DB
        hashed_baru = hash_password(password_baru)
        db.execute(
            text("UPDATE users SET password = :p WHERE id_user = :id"),
            {"p": hashed_baru, "id": id_user}
        )
        db.commit()

        return {"success": True, "message": "Password berhasil diubah!"}

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"Error change password: {e}")
        raise HTTPException(status_code=500, detail="Terjadi kesalahan pada server")