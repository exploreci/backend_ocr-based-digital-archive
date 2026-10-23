import re
from typing import Dict, Optional


def clean_noise(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'[^\w\s:/.-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    corrections = {
        "Unwra": "UNPRA",
        "Unpra": "UNPRA",
        "Universits": "Universitas",
        "Prabumulh": "Prabumulih",
        "Nornor": "Nomor",
        "Nomar": "Nomor",
    }
    for wrong, right in corrections.items():
        text = text.replace(wrong, right)

    return text.strip()


def normalize_text(raw_text: str):
    lines = [clean_noise(line) for line in raw_text.split("\n")]
    lines = [l for l in lines if l]
    return lines


def extract_metadata(raw_text: str, jenis_surat: Optional[str] = None) -> Dict[str, str]:

    lines = normalize_text(raw_text)

    # ── Selalu sediakan kedua field ──
    # Backend/Flutter tinggal ambil yang relevan
    data = {
        "asal_surat":    "",
        "tujuan_surat":  "",
        "nomor_surat":   "",
        "tanggal_surat": "",
        "perihal":       "",
    }

    if not lines:
        return data

    # ==========================================
    # 🟣 ASAL SURAT — surat MASUK
    # Ambil dari kop surat (baris-baris paling atas)
    # ==========================================
    if jenis_surat == "masuk":
        for line in lines[:10]:
            upper = line.upper()
            if any(x in upper for x in [
                "FAKULTAS", "PROGRAM", "UNIVERSITAS",
                "DINAS", "PEMERINTAH", "SEKOLAH", "LEMBAGA"
            ]):
                data["asal_surat"] = line
                break

        # Fallback: ambil baris pertama saja
        if not data["asal_surat"]:
            data["asal_surat"] = lines[0]

    # ==========================================
    # 🟠 TUJUAN SURAT — surat KELUAR
    # Cari kata kunci: Kepada, Yth, Yang Terhormat, To
    # ==========================================
    if jenis_surat == "keluar":

        for i, line in enumerate(lines):
            if re.search(r'\b(Kepada|Yth\.?|Yang\s+Terhormat|To)\b', line, re.IGNORECASE):

                isi = re.sub(
                    r'\b(Kepada\s*(Yth\.?)?|Yth\.?|Yang\s+Terhormat|To)\s*[:\-]?\s*',
                    '', line, flags=re.IGNORECASE
                ).strip()

                if len(isi) < 3 and i + 1 < len(lines):
                    isi = lines[i + 1]
                    if re.search(r'\b(Yth\.?)\b', isi, re.IGNORECASE):
                        isi = re.sub(
                            r'\b(Yth\.?)\s*[:\-]?\s*', '',
                            isi, flags=re.IGNORECASE
                        ).strip()
                        if len(isi) < 3 and i + 2 < len(lines):
                            isi = lines[i + 2]

                data["tujuan_surat"] = isi.strip(" :.-")

                # ── TAMBAHAN: cek baris-baris berikutnya ──
                # kalau ada list bernomor → kumpulkan semua, gabung dengan "|"
                tujuan_list = []

                # Kalau tujuan pertama sudah ada, masukkan dulu
                if data["tujuan_surat"]:
                    tujuan_list.append(data["tujuan_surat"])

                # Scan baris setelah posisi tujuan pertama ditemukan
                start = i + 1
                for j in range(start, min(start + 10, len(lines))):
                    baris = lines[j].strip()

                    # Stop kalau ketemu keyword surat lain
                    if re.search(
                        r'\b(Nomor|Perihal|Hal|Lampiran|Dari|Tanggal|Sifat)\b',
                        baris, re.IGNORECASE
                    ):
                        break

                    # Stop kalau baris kosong dan sudah kumpul > 1
                    if len(baris) == 0 and len(tujuan_list) > 1:
                        break

                    # Deteksi format bernomor: "1. xxx" atau "1) xxx"
                    match_nomor = re.match(r'^\d+[\.\)]\s*(.+)', baris)
                    if match_nomor:
                        kandidat = match_nomor.group(1).strip(" :.-")
                        # Hindari duplikat dengan tujuan pertama
                        if kandidat and kandidat not in tujuan_list:
                            tujuan_list.append(kandidat)
                        continue

                    # Deteksi format dash: "- xxx" atau "• xxx"
                    match_dash = re.match(r'^[-•]\s*(.+)', baris)
                    if match_dash:
                        kandidat = match_dash.group(1).strip(" :.-")
                        if kandidat and kandidat not in tujuan_list:
                            tujuan_list.append(kandidat)
                        continue

                # Kalau dapat lebih dari 1 tujuan → gabung dengan "|"
                if len(tujuan_list) > 1:
                    data["tujuan_surat"] = " | ".join(tujuan_list)

                break  # ← stop loop utama, sudah dapat tujuan

        # Fallback tetap sama persis seperti sebelumnya
        if not data["tujuan_surat"]:
            for i, line in enumerate(lines):
                if re.search(r'\bKepada\b', line, re.IGNORECASE):
                    for j in range(i + 1, min(i + 5, len(lines))):
                        candidate = lines[j].strip()
                        upper = candidate.upper()
                        if any(x in upper for x in [
                            "FAKULTAS", "PROGRAM", "UNIVERSITAS",
                            "DINAS", "PEMERINTAH", "BAPAK", "IBU",
                            "DIREKTUR", "REKTOR", "KEPALA", "KETUA"
                        ]):
                            data["tujuan_surat"] = candidate
                            break
                    break




    # ==========================================
    # 🔵 NOMOR SURAT — sama untuk masuk & keluar
    # ==========================================
    pola_keyword = r'(?:Nomor|No|Nornor|Nomar|N0|NO)'

    for line in lines:
        line_clean = line.strip()

        # Skip baris noise
        if any(x in line_clean.lower() for x in ["telp", "fax", "kode pos"]):
            continue

        match = re.search(
            rf'{pola_keyword}\s*[:.]?\s*([A-Z0-9./-]{{5,}})',
            line_clean, re.IGNORECASE
        )
        if match:
            kandidat = match.group(1).strip()
            if any(sym in kandidat for sym in ["/", "-", "."]):
                data["nomor_surat"] = kandidat.strip(" :.-")
                break

    # Fallback nomor surat
    if not data["nomor_surat"]:
        for line in lines:
            match = re.search(
                r'([A-Z0-9]{1,10}[/.-][A-Z0-9/.-]{5,})',
                line, re.IGNORECASE
            )
            if match:
                kandidat = match.group(1).strip()
                if any(sym in kandidat for sym in ["/", "-", "."]):
                    data["nomor_surat"] = kandidat.strip(" :.-")
                    break

    # ==========================================
    # 🟢 TANGGAL — sama untuk masuk & keluar
    # ==========================================
    bulan_map = {
        "januari": "01", "februari": "02", "maret": "03",
        "april": "04", "mei": "05", "juni": "06",
        "juli": "07", "agustus": "08", "september": "09",
        "oktober": "10", "november": "11", "desember": "12"
    }

    for line in lines:
        for bulan in bulan_map:
            if bulan in line.lower():
                match = re.search(
                    r'(\d{1,2})\s+' + bulan + r'\s+(\d{4})',
                    line, re.IGNORECASE
                )
                if match:
                    hari  = match.group(1).zfill(2)
                    tahun = match.group(2)
                    data["tanggal_surat"] = f"{tahun}-{bulan_map[bulan]}-{hari}"
                    break

    # ==========================================
    # 🟡 PERIHAL — sama untuk masuk & keluar
    # ==========================================
    for i, line in enumerate(lines):
        if re.search(r'(Perihal|Hal)', line, re.IGNORECASE):
            isi = re.sub(
                r'(Perihal|Hal)\s*[:\-]?\s*', '',
                line, flags=re.IGNORECASE
            )
            if len(isi) < 5 and i + 1 < len(lines):
                isi += " " + lines[i + 1]
            data["perihal"] = isi.strip()
            break

    return data






