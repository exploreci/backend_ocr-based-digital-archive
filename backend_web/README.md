# OCR-Based Digital Archive Backend

Sistem backend untuk aplikasi arsip digital dengan kemampuan OCR dan rule-based classification.

## Tech Stack
- Python 3.x
- FastAPI
- MySQL
- Tesseract OCR

## Setup & Run

## Setup & Run

1. Clone repo
2. Create virtual environment: `python -m venv .venv`
3. Activate: `.venv\Scripts\activate` (Windows) atau `source .venv/bin/activate` (Linux/Mac)
4. Install: `pip install -r requirements.txt`
5. Setup `.env` (copy dari `.env.example`)
6. Run: `uvicorn app:app --reload`
7. Server jalannya di `http://localhost:8000`
8. Swagger docs: `http://localhost:8000/docs`

## API Endpoints
- `POST /upload` - Upload dokumen
- `POST /process` - OCR & classification
- `GET /documents` - List dokumen
- `GET /documents/{id}` - Detail dokumen
- `POST /auth/login` - Login
- `POST /auth/register` - Register

*Dokumentasi lengkap: lihat di folder `/routes` atau jalankan server untuk akses Swagger docs di `/docs`*

## Features
- OCR menggunakan Tesseract
- Rule-based classification system
- Database MySQL untuk storage
- REST API dengan FastAPI