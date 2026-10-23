from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from routes.auth_routes import router as auth_router
from routes.arsip_routes import router as arsip_router
from routes.admin_routes import router as admin_router

from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

# include routes
app.include_router(auth_router)
# 2. Daftarkan router-nya di sini
app.include_router(arsip_router) # Sekarang pintu OCR & RBS sudah terbuka! 🚀
app.include_router(admin_router)


# Baris ini bikin folder 'uploads' bisa diakses via browser/Flutter
if not os.path.exists("uploads"):
    os.makedirs("uploads")

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# 🏠 TEST ROUTE
@app.get("/")
def home():
    return {
        "message": "Backend Arsip Digital aktif 🚀"
    }
    


# --- TAMBAHKAN INI ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Mengizinkan semua domain (termasuk Flutter Web kamu)
    allow_credentials=True,
    allow_methods=["*"],  # Mengizinkan POST, OPTIONS, GET, dll
    allow_headers=["*"],  # Mengizinkan semua header (Content-Type, Authorization, dll)
)
