# -*- coding: utf-8 -*-
"""
app.py — Banana AI · Klasifikasi Tingkat Kematangan Buah Pisang
================================================================
Aplikasi Streamlit satu-halaman (single-page dashboard) untuk Penulisan Ilmiah.

Model  : MobileNetV2 + Transfer Learning (base beku / feature extraction)
Input  : 224 x 224 RGB, normalisasi mobilenet_v2.preprocess_input ([-1, 1])
Kelas  : busuk, matang, mentah, terlalu_matang (dibaca dari class_indices.json)

CATATAN DESAIN
--------------
Berkas ini SELF-CONTAINED: seluruh logika pemuatan model & prediksi ditulis di
sini (tidak mengimpor utils/) agar deploy di Streamlit Community Cloud tidak
gagal karena masalah upload sub-modul. Seluruh angka performa dibaca dari folder
outputs/ pada saat runtime (bukan di-hardcode), sehingga selalu sinkron dengan
hasil pelatihan terbaru di notebook.
"""

from __future__ import annotations

import base64
import io
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

# --------------------------------------------------------------------------- #
# Identitas & path                                                            #
# --------------------------------------------------------------------------- #
STUDENT_NAME = "Rivan Ardi Nugroho"
STUDENT_NPM = "51423316"
STUDENT_CLASS = "3IA24"
PROGRAM_STUDI = "Informatika"
UNIVERSITAS = "Universitas Gunadarma"

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CLASS_INDICES_PATH = MODEL_DIR / "class_indices.json"
MODEL_FILENAMES = (
    "klasifikasi_kematangan_pisang_model.keras",
)
HERO_IMAGE_PATH = PROJECT_ROOT / "background_pisang.jpg"

IMAGE_SIZE: Tuple[int, int] = (224, 224)  # WAJIB sama dengan notebook training

# --------------------------------------------------------------------------- #
# Metadata kelas (label mockup + warna + narasi + urutan spektrum)            #
# Urutan spektrum kematangan: mentah -> matang -> terlalu_matang -> busuk      #
# --------------------------------------------------------------------------- #
CLASS_META: Dict[str, dict] = {
    "mentah": {
        "label": "Mentah",
        "en": "Unripe",
        "order": 0,
        "sphere": "radial-gradient(circle at 32% 28%, #86efac 0%, #22c55e 55%, #15803d 100%)",
        "color": "#16a34a",
        "soft": "#e7f6ec",
        "deskripsi": "Pisang mentah dengan warna kulit dominan hijau.",
        "rekomendasi": "Tunggu beberapa hari hingga matang sebelum dikonsumsi.",
        "penyimpanan": "Simpan di suhu ruang hingga kulit menguning.",
    },
    "matang": {
        "label": "Matang",
        "en": "Ripe",
        "order": 1,
        "sphere": "radial-gradient(circle at 32% 28%, #fde68a 0%, #f5c518 52%, #d19a12 100%)",
        "color": "#d19a12",
        "soft": "#fbf3d6",
        "deskripsi": "Pisang matang sempurna dengan warna kuning cerah.",
        "rekomendasi": "Siap dimakan langsung, rasa manis pada kondisi optimal.",
        "penyimpanan": "Simpan di suhu ruang, konsumsi dalam 1–2 hari.",
    },
    "terlalu_matang": {
        "label": "Terlalu Matang",
        "en": "Overripe",
        "order": 2,
        "sphere": "radial-gradient(circle at 32% 28%, #b08968 0%, #7c4a32 55%, #4a2c1a 100%)",
        "color": "#8a5a3b",
        "soft": "#f1e6dc",
        "deskripsi": "Pisang terlalu matang dengan banyak bintik cokelat.",
        "rekomendasi": "Cocok untuk smoothie, jus, atau banana bread.",
        "penyimpanan": "Segera konsumsi atau bekukan untuk penggunaan nanti.",
    },
    "busuk": {
        "label": "Busuk",
        "en": "Rotten",
        "order": 3,
        "sphere": "radial-gradient(circle at 32% 28%, #4b4453 0%, #2e2a33 55%, #17141b 100%)",
        "color": "#3a3340",
        "soft": "#e9e7ea",
        "deskripsi": "Pisang busuk, sudah tidak layak dikonsumsi.",
        "rekomendasi": "Jangan dikonsumsi, buang dengan cara yang benar.",
        "penyimpanan": "Tidak dapat disimpan.",
    },
}
SPECTRUM_ORDER = ["mentah", "matang", "terlalu_matang", "busuk"]

# Nilai fallback bila outputs/ tidak terbaca (tetap = data asli terakhir).
_FALLBACK_METRICS = {
    "accuracy": 0.9786,
    "loss": 0.0718,
    "baseline_acc": 0.1548,
    "per_class": {
        "busuk": {"precision": 0.9833, "recall": 0.9568, "f1": 0.9699, "support": 185},
        "matang": {"precision": 0.9809, "recall": 1.0000, "f1": 0.9904, "support": 154},
        "mentah": {"precision": 0.9730, "recall": 0.9818, "f1": 0.9774, "support": 110},
        "terlalu_matang": {"precision": 0.9737, "recall": 0.9823, "f1": 0.9780, "support": 113},
    },
    "split": {"train": 11793, "valid": 1123, "test": 562},
}

# --------------------------------------------------------------------------- #
# Konfigurasi halaman                                                          #
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Banana AI — Klasifikasi Kematangan Pisang",
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# --------------------------------------------------------------------------- #
# Utilitas kecil                                                              #
# --------------------------------------------------------------------------- #
def _fmt_id(n: int) -> str:
    """Format ribuan gaya Indonesia (titik): 13478 -> 13.478."""
    return f"{n:,}".replace(",", ".")


@lru_cache(maxsize=8)
def _img_data_uri(path_str: str) -> str:
    """Baca gambar dari disk -> data URI base64 (di-cache). '' bila gagal."""
    p = Path(path_str)
    if not p.exists():
        return ""
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"


def _encode_pil(image: Image.Image, max_side: int = 720) -> str:
    """PIL image -> data URI JPEG (diperkecil) untuk disematkan di kartu hasil."""
    disp = image.convert("RGB")
    disp.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    disp.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# --------------------------------------------------------------------------- #
# Membaca angka performa ASLI dari folder outputs/ (runtime, bukan hardcode)   #
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_report_metrics() -> dict:
    """
    Mem-parsing outputs/classification_report.txt & perbandingan_sebelum_sesudah.txt.
    Mengembalikan dict berisi accuracy, loss, baseline, metrik per-kelas, dan split.
    Jika berkas tidak ada / gagal di-parse, memakai nilai fallback (data asli terakhir).
    """
    metrics = json.loads(json.dumps(_FALLBACK_METRICS))  # deep copy

    report = OUTPUT_DIR / "classification_report.txt"
    if report.exists():
        try:
            text = report.read_text(encoding="utf-8", errors="ignore")
            m = re.search(r"Test accuracy:\s*([0-9.]+)\s*\|\s*Test loss:\s*([0-9.]+)", text)
            if m:
                metrics["accuracy"] = float(m.group(1))
                metrics["loss"] = float(m.group(2))
            for name in CLASS_META:
                row = re.search(
                    rf"^\s*{re.escape(name)}\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)",
                    text, flags=re.MULTILINE,
                )
                if row:
                    metrics["per_class"][name] = {
                        "precision": float(row.group(1)),
                        "recall": float(row.group(2)),
                        "f1": float(row.group(3)),
                        "support": int(row.group(4)),
                    }
        except Exception:
            pass

    comp = OUTPUT_DIR / "perbandingan_sebelum_sesudah.txt"
    if comp.exists():
        try:
            text = comp.read_text(encoding="utf-8", errors="ignore")
            b = re.search(r"Sebelum training\s+([0-9.]+)%", text)
            if b:
                metrics["baseline_acc"] = float(b.group(1)) / 100.0
        except Exception:
            pass

    # total test = jumlah support bila tersedia
    metrics["test_total"] = sum(v["support"] for v in metrics["per_class"].values())
    metrics["total_images"] = sum(metrics["split"].values())
    return metrics


# --------------------------------------------------------------------------- #
# Pemuatan model & prediksi (self-contained)                                  #
# --------------------------------------------------------------------------- #
class ModelNotFoundError(FileNotFoundError):
    pass


class InvalidImageError(ValueError):
    pass


def _find_model_path() -> Path:
    if not MODEL_DIR.exists():
        raise ModelNotFoundError(f"Folder model tidak ditemukan: {MODEL_DIR}")
    for name in MODEL_FILENAMES:
        cand = MODEL_DIR / name
        if cand.exists():
            return cand
    for pat in ("*.keras", "*.h5"):
        found = sorted(MODEL_DIR.glob(pat))
        if found:
            return found[0]
    raise ModelNotFoundError(f"Tidak ada berkas model (.keras/.h5) di: {MODEL_DIR}")


@st.cache_resource(show_spinner=False)
def get_artifacts():
    """Muat model Keras + pemetaan indeks->label. Di-cache sekali per sesi server."""
    with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
        class_indices = json.load(f)  # {nama_kelas: index}
    idx_to_label = {int(i): name for name, i in class_indices.items()}

    import tensorflow as tf  # impor lokal agar startup ringan
    model = tf.keras.models.load_model(_find_model_path())
    return model, idx_to_label


def _open_image(file) -> Image.Image:
    try:
        image = Image.open(file)
        image.load()
        return image
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError("File gambar tidak valid atau formatnya tidak didukung.") from exc


def _preprocess(image: Image.Image) -> np.ndarray:
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    image = image.convert("RGB").resize(IMAGE_SIZE)
    arr = np.asarray(image, dtype=np.float32)
    arr = preprocess_input(arr)              # normalisasi [-1, 1] (identik training)
    return np.expand_dims(arr, axis=0)


@st.cache_data(show_spinner=False)
def classify_bytes(file_bytes: bytes, _model, _idx_to_label) -> Tuple[str, float, Dict[str, float]]:
    """
    Prediksi satu citra dari bytes. Mengembalikan:
        (nama_kelas_teratas, confidence_0..1, {nama_kelas: probabilitas})
    bytes dipakai sebagai kunci cache (hashable); _model/_idx_to_label dikecualikan.
    """
    image = _open_image(io.BytesIO(file_bytes))
    probs = _model.predict(_preprocess(image), verbose=0)[0]
    prob_map = {_idx_to_label[i]: float(p) for i, p in enumerate(probs)}
    top_idx = int(np.argmax(probs))
    return _idx_to_label[top_idx], float(probs[top_idx]), prob_map


# --------------------------------------------------------------------------- #
# CSS (string biasa — bukan f-string — agar kurung kurawal tidak perlu di-escape)
# --------------------------------------------------------------------------- #
def inject_css() -> None:
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Manrope:wght@400;500;600;700&display=swap');

    :root{
      --ink:#1f2430; --ink-soft:#5b6472; --muted:#8b94a3;
      --amber:#f5a623; --amber-2:#f7931e; --gold:#f5b915;
      --grad:linear-gradient(135deg,#ffc93c 0%,#f7931e 100%);
      --grad-soft:linear-gradient(135deg,#ffd85e 0%,#f9a825 100%);
      --cream:#fffdf5; --card:#ffffff; --line:#eee6d2;
      --navy:#0f1420;
    }

    html{scroll-behavior:smooth;}
    .stApp{background:var(--cream);}
    #MainMenu, .stApp > footer, header[data-testid="stHeader"]{display:none!important;}
    section[data-testid="stSidebar"]{display:none!important;}
    [data-testid="stAppViewContainer"] > .main{padding:0;}
    .block-container{padding:0!important;max-width:100%!important;}
    html,body,[class*="css"]{font-family:'Manrope',system-ui,sans-serif;color:var(--ink);}

    /* ---------------- NAVBAR ---------------- */
    .nav{
      position:fixed;top:0;left:0;right:0;z-index:1000;height:68px;
      display:flex;align-items:center;justify-content:space-between;
      padding:0 clamp(18px,5vw,72px);
      background:rgba(255,253,245,.86);backdrop-filter:blur(12px);
      border-bottom:1px solid var(--line);
    }
    .nav-brand{display:flex;align-items:center;gap:12px;}
    .nav-logo{
      width:42px;height:42px;border-radius:13px;background:var(--grad);
      display:flex;align-items:center;justify-content:center;font-size:22px;
      box-shadow:0 6px 16px rgba(247,147,30,.32);
    }
    .nav-name{font-family:'Plus Jakarta Sans';font-weight:800;font-size:1.32rem;
      background:var(--grad);-webkit-background-clip:text;background-clip:text;
      -webkit-text-fill-color:transparent;}
    .nav-links{display:flex;align-items:center;gap:34px;}
    .nav-links a{color:var(--ink);text-decoration:none;font-weight:600;font-size:.95rem;
      transition:color .18s;}
    .nav-links a:hover{color:var(--amber-2);}
    .nav-cta{
      background:var(--grad);color:#fff!important;padding:11px 26px;border-radius:999px;
      font-weight:700!important;box-shadow:0 8px 20px rgba(247,147,30,.34);
    }
    .nav-cta:hover{color:#fff!important;filter:brightness(1.03);}
    @media(max-width:860px){.nav-links a:not(.nav-cta){display:none;}}

    /* ---------------- LAYOUT SECTIONS ---------------- */
    .section{padding:96px clamp(18px,6vw,80px);max-width:1180px;margin:0 auto;}
    .section-lg{max-width:1240px;}
    [id]{scroll-margin-top:84px;}
    .eyebrow{font-family:'Plus Jakarta Sans';font-weight:800;letter-spacing:.22em;
      text-transform:uppercase;font-size:.76rem;color:var(--amber-2);text-align:center;}
    .h2{font-family:'Plus Jakarta Sans';font-weight:800;font-size:clamp(2rem,4vw,2.9rem);
      text-align:center;margin:.5rem 0 0;letter-spacing:-.02em;line-height:1.1;}
    .h2 .g{background:var(--grad);-webkit-background-clip:text;background-clip:text;
      -webkit-text-fill-color:transparent;}
    .lead{max-width:640px;margin:1.1rem auto 0;text-align:center;color:var(--ink-soft);
      font-size:1.06rem;line-height:1.6;}
    /* Menang atas CSS bawaan Streamlit untuk <p> (specificity lebih tinggi)
       agar teks keterangan benar-benar center & tidak melebar rata-kiri. */
    [data-testid="stMarkdownContainer"] p.lead{
      text-align:center!important;max-width:640px!important;
      margin-left:auto!important;margin-right:auto!important;}
    [data-testid="stMarkdownContainer"] .eyebrow,
    [data-testid="stMarkdownContainer"] .h2{text-align:center!important;}

    /* ---------------- HERO ---------------- */
    .hero{
      position:relative;margin-top:68px;padding:70px clamp(18px,6vw,80px) 90px;
      background:
        radial-gradient(760px 460px at 18% 8%, rgba(255,214,102,.55) 0%, rgba(255,214,102,0) 60%),
        radial-gradient(720px 520px at 88% 30%, rgba(249,168,37,.30) 0%, rgba(249,168,37,0) 62%),
        var(--cream);
      overflow:hidden;
    }
    .hero-grid{max-width:1180px;margin:0 auto;display:grid;
      grid-template-columns:1.05fr .95fr;gap:56px;align-items:center;}
    .hero-badge{display:inline-flex;align-items:center;gap:8px;font-weight:700;
      font-size:.82rem;color:var(--amber-2);background:rgba(247,147,30,.12);
      padding:8px 16px;border-radius:999px;}
    .hero-title{font-family:'Plus Jakarta Sans';font-weight:800;
      font-size:clamp(2.6rem,5.6vw,4.3rem);line-height:1.02;letter-spacing:-.03em;
      margin:20px 0 0;}
    .hero-title .g{background:var(--grad);-webkit-background-clip:text;
      background-clip:text;-webkit-text-fill-color:transparent;}
    .hero-sub{color:var(--ink-soft);font-size:1.12rem;line-height:1.6;
      margin:22px 0 0;max-width:30rem;}
    .hero-actions{display:flex;gap:14px;margin-top:32px;flex-wrap:wrap;}
    .btn-primary{background:var(--grad);color:#fff;font-weight:700;border:0;
      padding:15px 30px;border-radius:14px;text-decoration:none;font-size:1rem;
      display:inline-flex;align-items:center;gap:10px;
      box-shadow:0 12px 26px rgba(247,147,30,.34);transition:transform .14s,filter .2s;}
    .btn-primary:hover{transform:translateY(-2px);filter:brightness(1.03);}
    .btn-ghost{background:#fff;color:var(--ink);font-weight:700;border:1.6px solid var(--line);
      padding:15px 30px;border-radius:14px;text-decoration:none;font-size:1rem;
      display:inline-flex;align-items:center;gap:10px;transition:border-color .2s,transform .14s;}
    .btn-ghost:hover{border-color:var(--amber);transform:translateY(-2px);}
    .hero-stats{display:flex;gap:44px;margin-top:46px;}
    .hero-stat .num{font-family:'Plus Jakarta Sans';font-weight:800;font-size:2.5rem;
      background:var(--grad);-webkit-background-clip:text;background-clip:text;
      -webkit-text-fill-color:transparent;line-height:1;}
    .hero-stat .cap{color:var(--muted);font-size:.85rem;font-weight:600;margin-top:6px;}
    .hero-media{position:relative;}
    .hero-media img{width:100%;border-radius:26px;object-fit:cover;aspect-ratio:4/4.4;
      box-shadow:0 30px 70px rgba(31,36,48,.20);}
    .hero-media::after{content:"";position:absolute;inset:-14px;z-index:-1;border-radius:34px;
      background:var(--grad);opacity:.16;filter:blur(24px);}
    @media(max-width:900px){.hero-grid{grid-template-columns:1fr;gap:36px;}
      .hero-media{order:-1;} .hero-stats{gap:28px;}}

    /* ---------------- CARDS ---------------- */
    .grid-3{display:grid;grid-template-columns:repeat(3,1fr);gap:24px;margin-top:52px;}
    .grid-2{display:grid;grid-template-columns:.9fr 1.1fr;gap:34px;margin-top:52px;align-items:stretch;}
    @media(max-width:860px){.grid-3{grid-template-columns:1fr;}.grid-2{grid-template-columns:1fr;}}
    .card{background:var(--card);border:1px solid var(--line);border-radius:22px;
      padding:34px 30px;box-shadow:0 10px 34px rgba(31,36,48,.05);transition:transform .16s,box-shadow .2s;}
    .card:hover{transform:translateY(-4px);box-shadow:0 20px 46px rgba(31,36,48,.10);}
    .ic{width:66px;height:66px;border-radius:18px;display:flex;align-items:center;
      justify-content:center;font-size:30px;color:#fff;margin-bottom:20px;}
    .card h3{font-family:'Plus Jakarta Sans';font-weight:700;font-size:1.28rem;margin:0 0 10px;}
    .card p{color:var(--ink-soft);font-size:.98rem;line-height:1.6;margin:0;}
    .pill{display:inline-block;margin-top:18px;font-weight:700;font-size:.8rem;
      padding:7px 16px;border-radius:999px;}

    /* ABOUT mission card + ripeness mini cards */
    .mission{background:var(--card);border:1px solid var(--line);border-radius:24px;
      padding:38px 34px;box-shadow:0 12px 36px rgba(31,36,48,.06);}
    .mission h3{font-family:'Plus Jakarta Sans';font-weight:800;font-size:1.7rem;margin:18px 0 14px;}
    .mission p{color:var(--ink-soft);line-height:1.65;margin:0 0 22px;}
    .check{display:flex;align-items:center;gap:12px;margin:12px 0;font-weight:600;color:var(--ink);}
    .check .tick{width:26px;height:26px;border-radius:999px;background:var(--grad);color:#fff;
      display:flex;align-items:center;justify-content:center;font-size:14px;flex:0 0 auto;}
    .ripe-grid{display:grid;grid-template-columns:1fr 1fr;gap:20px;}
    .ripe{background:var(--card);border:1px solid var(--line);border-radius:20px;padding:24px;
      box-shadow:0 8px 24px rgba(31,36,48,.05);}
    .sphere{width:58px;height:58px;border-radius:999px;margin-bottom:16px;
      box-shadow:0 8px 18px rgba(0,0,0,.18);}
    .ripe h4{font-family:'Plus Jakarta Sans';font-weight:700;font-size:1.12rem;margin:0 0 6px;}
    .ripe p{color:var(--ink-soft);font-size:.9rem;line-height:1.5;margin:0;}

    /* Architecture strip */
    .arch{background:var(--card);border:1px solid var(--line);border-radius:24px;
      padding:40px;margin-top:34px;box-shadow:0 12px 36px rgba(31,36,48,.05);}
    .arch h3{font-family:'Plus Jakarta Sans';font-weight:800;font-size:1.6rem;text-align:center;margin:0 0 34px;}
    .arch-row{display:grid;grid-template-columns:repeat(4,1fr);gap:20px;}
    @media(max-width:860px){.arch-row{grid-template-columns:1fr 1fr;}}
    .arch-step{text-align:center;}
    .arch-ic{width:64px;height:64px;border-radius:18px;margin:0 auto 14px;color:#fff;font-size:26px;
      display:flex;align-items:center;justify-content:center;}
    .arch-step b{font-family:'Plus Jakarta Sans';font-size:1.05rem;display:block;}
    .arch-step span{color:var(--muted);font-size:.85rem;}

    /* Cara Kerja steps */
    .steps{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:56px;position:relative;}
    @media(max-width:860px){.steps{grid-template-columns:1fr 1fr;gap:30px 8px;}}
    .step{text-align:center;padding:0 10px;}
    .step-num{width:66px;height:66px;border-radius:999px;background:var(--grad);color:#fff;
      font-family:'Plus Jakarta Sans';font-weight:800;font-size:1.5rem;margin:0 auto 18px;
      display:flex;align-items:center;justify-content:center;
      box-shadow:0 10px 22px rgba(247,147,30,.34);}
    .step b{font-family:'Plus Jakarta Sans';font-size:1.12rem;display:block;margin-bottom:8px;}
    .step p{color:var(--ink-soft);font-size:.92rem;line-height:1.5;margin:0;}

    /* ---------------- GET STARTED ---------------- */
    .try-band{background:linear-gradient(180deg,#fffaf0 0%,#fff6e2 100%);}

    /* Kartu pembungkus (via st.container(key="uploadcard")) */
    .st-key-uploadcard{max-width:820px;margin:36px auto 0;background:var(--card);
      border:1px solid var(--line);border-radius:26px;padding:40px clamp(22px,4vw,48px);
      box-shadow:0 20px 60px rgba(31,36,48,.10);}
    .uploadcard-title{font-family:'Plus Jakarta Sans';font-weight:800;font-size:1.55rem;
      text-align:center;margin:0 0 24px;display:flex;align-items:center;justify-content:center;gap:12px;}

    /* Native file_uploader -> dropzone gaya mockup (Streamlit 1.59 DOM) */
    [data-testid="stFileUploader"]{margin:0;}
    [data-testid="stFileUploader"] label{display:none!important;}
    section[data-testid="stFileUploaderDropzone"]{
      display:flex;flex-direction:column;align-items:center;justify-content:center;
      text-align:center;gap:8px;min-height:250px;padding:34px 20px;cursor:pointer;
      border:2px dashed var(--gold);border-radius:20px;background:#fffdf7;
      transition:border-color .2s,background .2s;}
    section[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--amber-2);background:#fff9ec;}
    /* ikon awan bulat di atas */
    section[data-testid="stFileUploaderDropzone"]::before{
      content:"☁️";order:-3;width:84px;height:84px;border-radius:999px;background:var(--grad);
      display:flex;align-items:center;justify-content:center;font-size:38px;
      box-shadow:0 10px 24px rgba(247,147,30,.34);margin-bottom:6px;}
    /* dua baris instruksi kustom (teks bawaan disembunyikan) */
    [data-testid="stFileUploaderDropzoneInstructions"]{order:-1;display:flex;
      flex-direction:column;align-items:center;gap:4px;}
    [data-testid="stFileUploaderDropzoneInstructions"] *{display:none!important;}
    [data-testid="stFileUploaderDropzoneInstructions"]::before{
      content:"Klik atau drag & drop gambar";font-family:'Plus Jakarta Sans';
      font-weight:700;font-size:1.18rem;color:var(--ink);}
    [data-testid="stFileUploaderDropzoneInstructions"]::after{
      content:"PNG, JPG, JPEG · maks 16MB";color:var(--muted);font-size:.92rem;}
    /* tombol bawaan uploader -> pill "Pilih atau Ambil Foto"
       PENTING: di-scope ke dalam dropzone agar TIDAK mengenai tombol Analisis
       (st.button juga memakai data-testid="stBaseButton-secondary"). */
    section[data-testid="stFileUploaderDropzone"] > span{order:2;margin-top:6px;}
    section[data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"]{
      background:var(--grad)!important;color:#fff!important;border:0!important;
      border-radius:999px!important;padding:10px 24px!important;font-weight:700!important;
      box-shadow:0 8px 18px rgba(247,147,30,.3)!important;}
    /* sembunyikan seluruh isi bawaan (ikon + teks "Upload"), ganti dengan label kustom */
    section[data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"] > div > *{
      display:none!important;}
    section[data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"] > div::after{
      content:"🖼️  Pilih atau Ambil Foto";font-size:.95rem;font-weight:700;color:#fff;}
    /* daftar file terunggah (state setelah upload) dirapikan */
    [data-testid="stFileUploaderFile"]{color:var(--ink-soft);}
    /* saat file sudah dipilih (elemen instruksi hilang): sembunyikan ikon awan
       + rapatkan dropzone. Pakai :not(:has(instruksi)) agar andal lintas versi. */
    section[data-testid="stFileUploaderDropzone"]:not(:has([data-testid="stFileUploaderDropzoneInstructions"])){
      min-height:auto;padding:18px 20px;}
    section[data-testid="stFileUploaderDropzone"]:not(:has([data-testid="stFileUploaderDropzoneInstructions"]))::before{
      display:none!important;}

    /* Tombol analisis (satu-satunya st.button di halaman) */
    div.stButton > button{
      width:100%;background:var(--grad);color:#fff!important;border:0;border-radius:16px;
      padding:16px 20px;font-family:'Plus Jakarta Sans';font-weight:800;font-size:1.14rem;
      letter-spacing:.01em;box-shadow:0 14px 30px rgba(247,147,30,.32);
      transition:transform .14s,filter .2s;}
    /* Paksa SEMUA teks/isi di dalam tombol jadi putih (menang atas gaya bawaan Streamlit)
       supaya "Analisis Kematangan Sekarang" + ikon otak tampil jelas & kontras. */
    div.stButton > button p,
    div.stButton > button div,
    div.stButton > button span{color:#fff!important;}
    div.stButton > button *{opacity:1!important;}
    div.stButton > button:hover{transform:translateY(-2px);filter:brightness(1.03);color:#fff!important;}
    div.stButton > button:focus{color:#fff!important;box-shadow:0 0 0 3px rgba(247,147,30,.35);}
    /* State nonaktif (belum ada gambar): teks tetap putih & terbaca, hanya diredupkan. */
    div.stButton > button:disabled{opacity:.6;}
    div.stButton > button:disabled p,
    div.stButton > button:disabled div,
    div.stButton > button:disabled span{color:#fff!important;}
    /* Ikon otak PUTIH via SVG mask (tegas & konsisten di semua browser,
       menggantikan emoji yang warnanya bergantung OS). */
    div.stButton > button{display:flex!important;align-items:center;justify-content:center;gap:10px;}
    /* Cegah wrapper label bawaan Streamlit melebar (grow) — supaya ikon + teks
       menyatu sebagai satu grup dan benar-benar center, bukan ikon terdorong ke tepi. */
    div.stButton > button > *{flex:0 0 auto!important;width:auto!important;}
    div.stButton > button::before{content:"";flex:0 0 auto;width:22px;height:22px;
      background-color:#fff;
      -webkit-mask:url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA1MTIgNTEyIj48cGF0aCBkPSJNMTg0IDBjMzAuOSAwIDU2IDI1LjEgNTYgNTZWNDU2YzAgMzAuOS0yNS4xIDU2LTU2IDU2Yy0yOC45IDAtNTIuNy0yMS45LTU1LjctNTAuMWMtNS4yIDEuNC0xMC43IDIuMS0xNi4zIDIuMWMtMzUuMyAwLTY0LTI4LjctNjQtNjRjMC03LjQgMS4zLTE0LjYgMy42LTIxLjJDMjEuNCAzNjcuNCAwIDMzOC4yIDAgMzA0YzAtMzEuOSAxOC43LTU5LjUgNDUuOC03Mi4zQzM3LjEgMjIwLjggMzIgMjA3IDMyIDE5MmMwLTMwLjcgMjEuNi01Ni4zIDUwLjQtNjIuNkM4MC44IDEyMy45IDgwIDExOCA4MCAxMTJjMC0yOS45IDIwLjYtNTUuMSA0OC4zLTYyLjFDMTMxLjMgMjEuOSAxNTUuMSAwIDE4NCAwek0zMjggMGMyOC45IDAgNTIuNiAyMS45IDU1LjcgNDkuOUM0MTEuNCA1Ni45IDQzMiA4Mi4xIDQzMiAxMTJjMCA2LS44IDExLjktMi40IDE3LjRDNDU4LjQgMTM1LjcgNDgwIDE2MS4zIDQ4MCAxOTJjMCAxNS01LjEgMjguOC0xMy44IDM5LjdDNDkzLjMgMjQ0LjUgNTEyIDI3Mi4xIDUxMiAzMDRjMCAzNC4yLTIxLjQgNjMuNC01MS42IDc0LjhjMi4zIDYuNiAzLjYgMTMuOCAzLjYgMjEuMmMwIDM1LjMtMjguNyA2NC02NCA2NGMtNS42IDAtMTEuMS0uNy0xNi4zLTIuMWMtMyAyOC4yLTI2LjggNTAuMS01NS43IDUwLjFjLTMwLjkgMC01Ni0yNS4xLTU2LTU2VjU2YzAtMzAuOSAyNS4xLTU2IDU2LTU2eiIvPjwvc3ZnPg==) center/contain no-repeat;
      mask:url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA1MTIgNTEyIj48cGF0aCBkPSJNMTg0IDBjMzAuOSAwIDU2IDI1LjEgNTYgNTZWNDU2YzAgMzAuOS0yNS4xIDU2LTU2IDU2Yy0yOC45IDAtNTIuNy0yMS45LTU1LjctNTAuMWMtNS4yIDEuNC0xMC43IDIuMS0xNi4zIDIuMWMtMzUuMyAwLTY0LTI4LjctNjQtNjRjMC03LjQgMS4zLTE0LjYgMy42LTIxLjJDMjEuNCAzNjcuNCAwIDMzOC4yIDAgMzA0YzAtMzEuOSAxOC43LTU5LjUgNDUuOC03Mi4zQzM3LjEgMjIwLjggMzIgMjA3IDMyIDE5MmMwLTMwLjcgMjEuNi01Ni4zIDUwLjQtNjIuNkM4MC44IDEyMy45IDgwIDExOCA4MCAxMTJjMC0yOS45IDIwLjYtNTUuMSA0OC4zLTYyLjFDMTMxLjMgMjEuOSAxNTUuMSAwIDE4NCAwek0zMjggMGMyOC45IDAgNTIuNiAyMS45IDU1LjcgNDkuOUM0MTEuNCA1Ni45IDQzMiA4Mi4xIDQzMiAxMTJjMCA2LS44IDExLjktMi40IDE3LjRDNDU4LjQgMTM1LjcgNDgwIDE2MS4zIDQ4MCAxOTJjMCAxNS01LjEgMjguOC0xMy44IDM5LjdDNDkzLjMgMjQ0LjUgNTEyIDI3Mi4xIDUxMiAzMDRjMCAzNC4yLTIxLjQgNjMuNC01MS42IDc0LjhjMi4zIDYuNiAzLjYgMTMuOCAzLjYgMjEuMmMwIDM1LjMtMjguNyA2NC02NCA2NGMtNS42IDAtMTEuMS0uNy0xNi4zLTIuMWMtMyAyOC4yLTI2LjggNTAuMS01NS43IDUwLjFjLTMwLjkgMC01Ni0yNS4xLTU2LTU2VjU2YzAtMzAuOSAyNS4xLTU2IDU2LTU2eiIvPjwvc3ZnPg==) center/contain no-repeat;}

    [data-testid="stImage"] img{border-radius:16px;}

    /* Loader pisang berputar */
    .banana-loader{display:flex;flex-direction:column;align-items:center;gap:16px;
      padding:38px 0 26px;}
    .banana-spin{font-size:62px;line-height:1;animation:spin 1s linear infinite;
      filter:drop-shadow(0 8px 14px rgba(247,147,30,.4));}
    @keyframes spin{to{transform:rotate(360deg);}}
    @media(prefers-reduced-motion:reduce){.banana-spin{animation:spin 2.4s linear infinite;}}
    .banana-loader .txt{font-family:'Plus Jakarta Sans';font-weight:700;color:var(--amber-2);}

    /* Kartu hasil */
    .result{margin-top:30px;background:var(--card);border:1px solid var(--line);
      border-radius:24px;padding:38px clamp(20px,4vw,44px);box-shadow:0 16px 46px rgba(31,36,48,.08);}
    .result .rsphere{width:96px;height:96px;border-radius:999px;margin:0 auto 18px;
      box-shadow:0 12px 26px rgba(0,0,0,.22);}
    .result .rname{font-family:'Plus Jakarta Sans';font-weight:800;font-size:2rem;text-align:center;margin:0;}
    .result .rconf{display:block;width:max-content;margin:16px auto 0;background:var(--grad);
      color:#fff;font-weight:800;padding:9px 26px;border-radius:999px;font-size:1rem;
      box-shadow:0 10px 22px rgba(247,147,30,.3);}
    .rfield{margin-top:26px;}
    .rfield .lab{font-family:'Plus Jakarta Sans';font-weight:700;font-size:1rem;
      display:flex;align-items:center;gap:9px;margin-bottom:6px;}
    .rfield .val{color:var(--ink-soft);line-height:1.6;}
    .rprob{margin-top:30px;}
    .rprob > .lab{font-family:'Plus Jakarta Sans';font-weight:700;font-size:1rem;
      display:flex;align-items:center;gap:9px;margin-bottom:16px;}
    .prow{margin:16px 0;}
    .prow .t{display:flex;justify-content:space-between;font-size:.94rem;margin-bottom:6px;}
    .prow .t b{font-family:'Plus Jakarta Sans';font-weight:700;}
    .prob-track{height:9px;border-radius:999px;background:#efe7d4;overflow:hidden;}
    .prob-fill{height:100%;border-radius:999px;transition:width .5s ease;}
    .disclaimer{margin-top:26px;padding-top:18px;border-top:1px solid var(--line);
      font-size:.85rem;color:var(--muted);line-height:1.55;}

    /* ---------------- FOOTER ---------------- */
    .footer{background:
      radial-gradient(600px 300px at 50% 0%, rgba(247,147,30,.16) 0%, rgba(247,147,30,0) 70%),
      var(--navy);color:#cbd2df;padding:64px clamp(18px,6vw,80px) 30px;}
    .footer-grid{max-width:1180px;margin:0 auto;display:grid;
      grid-template-columns:1.6fr 1fr 1fr 1.2fr;gap:38px;}
    @media(max-width:860px){.footer-grid{grid-template-columns:1fr 1fr;gap:30px;}}
    .footer-brand{display:flex;align-items:center;gap:12px;margin-bottom:16px;}
    .footer-brand .nav-logo{width:40px;height:40px;font-size:20px;}
    .footer-brand b{font-family:'Plus Jakarta Sans';font-weight:800;font-size:1.3rem;color:#fff;}
    .footer p{font-size:.92rem;line-height:1.65;color:#9aa4b5;margin:0;max-width:22rem;}
    .footer h5{font-family:'Plus Jakarta Sans';font-weight:700;color:#fff;font-size:1rem;margin:0 0 16px;}
    .footer a{color:#9aa4b5;text-decoration:none;display:block;margin:9px 0;font-size:.92rem;transition:color .18s;}
    .footer a:hover{color:var(--gold);}
    .fcontact{display:flex;align-items:center;gap:10px;margin:9px 0;font-size:.92rem;color:#9aa4b5;}
    .ftick{color:var(--gold);}
    .footer-bottom{max-width:1180px;margin:40px auto 0;padding-top:22px;
      border-top:1px solid rgba(255,255,255,.09);text-align:center;font-size:.85rem;color:#7c8496;}
    .footer-bottom b{color:#cbd2df;}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Komponen render (HTML statis via st.markdown)                               #
# --------------------------------------------------------------------------- #
def render_navbar() -> None:
    st.markdown(
        """
        <div class="nav">
          <div class="nav-brand">
            <div class="nav-logo">🍌</div>
            <div class="nav-name">Banana AI</div>
          </div>
          <div class="nav-links">
            <a href="#home">Home</a>
            <a href="#about">About</a>
            <a href="#technology">Technology</a>
            <a href="#features">Features</a>
            <a href="#get-started" class="nav-cta">Get Started</a>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero(metrics: dict) -> None:
    acc = metrics["accuracy"] * 100
    total = _fmt_id(metrics["total_images"])
    hero_img = _img_data_uri(str(HERO_IMAGE_PATH))
    img_tag = (
        f'<img src="{hero_img}" alt="Buah pisang">'
        if hero_img
        else '<div style="aspect-ratio:4/4.4;border-radius:26px;background:var(--grad-soft);"></div>'
    )
    st.markdown(
        f"""
        <section id="home" class="hero">
          <div class="hero-grid">
            <div>
              <span class="hero-badge">🍌 AI-Powered Classification</span>
              <h1 class="hero-title">Klasifikasi <span class="g">Kematangan Pisang</span> dengan AI</h1>
              <p class="hero-sub">Sistem cerdas berbasis Deep Learning untuk mengidentifikasi
                tingkat kematangan pisang secara akurat dan real-time.</p>
              <div class="hero-actions">
                <a href="#get-started" class="btn-primary">⬆ Coba Sekarang</a>
                <a href="#about" class="btn-ghost">▶ Pelajari Lebih</a>
              </div>
              <div class="hero-stats">
                <div class="hero-stat"><div class="num">{acc:.2f}</div><div class="cap">Akurasi&nbsp;%</div></div>
                <div class="hero-stat"><div class="num">4</div><div class="cap">Kelas Kematangan</div></div>
                <div class="hero-stat"><div class="num">{total}</div><div class="cap">Total Citra</div></div>
              </div>
            </div>
            <div class="hero-media">{img_tag}</div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_about(metrics: dict) -> None:
    acc = metrics["accuracy"] * 100
    ripe_cards = ""
    for key in SPECTRUM_ORDER:
        m = CLASS_META[key]
        ripe_cards += (
            f'<div class="ripe"><div class="sphere" style="background:{m["sphere"]};"></div>'
            f'<h4>{m["en"]}</h4><p>{m["deskripsi"]}</p></div>'
        )
    st.markdown(
        f"""
        <section id="about" class="section">
          <div class="eyebrow">About System</div>
          <h2 class="h2">Tentang <span class="g">Banana AI</span></h2>
          <p class="lead">Sistem klasifikasi otomatis yang memanfaatkan teknologi Deep Learning
            untuk mengidentifikasi tingkat kematangan pisang dengan presisi tinggi.</p>
          <div class="grid-2">
            <div class="mission">
              <div class="ic" style="background:var(--grad);">🎯</div>
              <h3>Misi Kami</h3>
              <p>Mengembangkan solusi AI yang membantu industri pertanian dan distribusi buah
                dalam menentukan kualitas dan kematangan pisang secara objektif dan efisien.</p>
              <div class="check"><span class="tick">✓</span> Akurasi tinggi {acc:.2f}% pada data uji</div>
              <div class="check"><span class="tick">✓</span> Hasil klasifikasi real-time</div>
              <div class="check"><span class="tick">✓</span> Antarmuka mudah digunakan</div>
            </div>
            <div class="ripe-grid">{ripe_cards}</div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_technology() -> None:
    st.markdown(
        """
        <section id="technology" class="section">
          <div class="eyebrow">Technology Stack</div>
          <h2 class="h2">Teknologi <span class="g">Canggih</span></h2>
          <p class="lead">Dibangun dengan teknologi terkini dalam bidang Machine Learning
            dan Web Development.</p>
          <div class="grid-3">
            <div class="card">
              <div class="ic" style="background:linear-gradient(135deg,#ff7a59,#ff5252);">🧠</div>
              <h3>TensorFlow / Keras</h3>
              <p>Framework Deep Learning untuk membangun dan melatih model Convolutional
                Neural Network (CNN) dengan transfer learning.</p>
              <span class="pill" style="background:#ffe9e3;color:#e5533c;">Deep Learning</span>
            </div>
            <div class="card">
              <div class="ic" style="background:linear-gradient(135deg,#f5b915,#f7931e);">🚀</div>
              <h3>Streamlit</h3>
              <p>Framework Python untuk membangun antarmuka web interaktif dan mem-deploy
                model ke Streamlit Community Cloud tanpa server terpisah.</p>
              <span class="pill" style="background:#fdf0d6;color:#c98a10;">Web App</span>
            </div>
            <div class="card">
              <div class="ic" style="background:linear-gradient(135deg,#ffb020,#ff8a00);">🗂️</div>
              <h3>MobileNetV2 (CNN)</h3>
              <p>Arsitektur Convolutional Neural Network ringan dengan transfer learning
                untuk ekstraksi fitur visual pisang secara efisien.</p>
              <span class="pill" style="background:#fff3cf;color:#c99a10;">AI Model</span>
            </div>
          </div>

          <div class="arch">
            <h3>Arsitektur Model</h3>
            <div class="arch-row">
              <div class="arch-step">
                <div class="arch-ic" style="background:linear-gradient(135deg,#4f9dff,#2f6fe0);">🖼️</div>
                <b>Input Layer</b><span>224 × 224 RGB</span>
              </div>
              <div class="arch-step">
                <div class="arch-ic" style="background:linear-gradient(135deg,#a06bff,#7c3aed);">🔺</div>
                <b>MobileNetV2 Base</b><span>Feature Extraction (beku)</span>
              </div>
              <div class="arch-step">
                <div class="arch-ic" style="background:linear-gradient(135deg,#34d27b,#16a34a);">➕</div>
                <b>Pooling + Dropout</b><span>GAP · Dropout 0.5</span>
              </div>
              <div class="arch-step">
                <div class="arch-ic" style="background:linear-gradient(135deg,#ffb020,#f7931e);">🎯</div>
                <b>Dense (Softmax)</b><span>4 Kelas Output</span>
              </div>
            </div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_features(metrics: dict) -> None:
    acc = metrics["accuracy"] * 100
    st.markdown(
        f"""
        <section id="features" class="section">
          <div class="eyebrow">Features</div>
          <h2 class="h2">Fitur <span class="g">Unggulan</span></h2>
          <p class="lead">Berbagai fitur yang membuat sistem ini menjadi solusi praktis
            untuk klasifikasi kematangan pisang.</p>
          <div class="grid-3">
            <div class="card"><div class="ic" style="background:linear-gradient(135deg,#ffb020,#f7931e);">⚡</div>
              <h3>Real-time Processing</h3><p>Proses analisis citra dalam hitungan detik dengan hasil yang akurat dan detail.</p></div>
            <div class="card"><div class="ic" style="background:linear-gradient(135deg,#4f9dff,#2f6fe0);">📈</div>
              <h3>High Accuracy</h3><p>Model terlatih pada data uji mencapai akurasi {acc:.2f}% dalam klasifikasi empat kelas.</p></div>
            <div class="card"><div class="ic" style="background:linear-gradient(135deg,#34d27b,#16a34a);">📱</div>
              <h3>Responsive Design</h3><p>Antarmuka responsif yang dapat diakses dari perangkat mobile maupun desktop.</p></div>
            <div class="card"><div class="ic" style="background:linear-gradient(135deg,#a06bff,#7c3aed);">💡</div>
              <h3>Detailed Insights</h3><p>Informasi lengkap: deskripsi kondisi, rekomendasi, cara penyimpanan, dan probabilitas tiap kelas.</p></div>
            <div class="card"><div class="ic" style="background:linear-gradient(135deg,#ff6b6b,#e5533c);">🛡️</div>
              <h3>Secure &amp; Private</h3><p>Gambar diproses saat itu juga dan tidak disimpan permanen di server.</p></div>
            <div class="card"><div class="ic" style="background:linear-gradient(135deg,#ff7ab6,#d6318f);">👥</div>
              <h3>Easy to Use</h3><p>Antarmuka intuitif yang mudah digunakan bahkan untuk pengguna pertama kali.</p></div>
          </div>

          <div class="steps">
            <div class="step"><div class="step-num">1</div><b>Upload Gambar</b><p>Pilih atau seret foto pisang yang ingin dianalisis.</p></div>
            <div class="step"><div class="step-num">2</div><b>AI Analysis</b><p>Model CNN menganalisis fitur visual gambar.</p></div>
            <div class="step"><div class="step-num">3</div><b>Classification</b><p>Sistem mengklasifikasikan tingkat kematangan.</p></div>
            <div class="step"><div class="step-num">4</div><b>Get Results</b><p>Terima hasil detail beserta rekomendasi.</p></div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_result_card(top_key: str, confidence: float, prob_map: Dict[str, float]) -> None:
    m = CLASS_META[top_key]
    conf = confidence * 100
    prob_rows = ""
    for key in SPECTRUM_ORDER:
        p = prob_map.get(key, 0.0) * 100
        cm = CLASS_META[key]
        prob_rows += (
            f'<div class="prow"><div class="t"><span>{cm["label"]} '
            f'<em style="color:var(--muted);font-style:normal;">({cm["en"]})</em></span>'
            f'<b>{p:.2f}%</b></div>'
            f'<div class="prob-track"><div class="prob-fill" '
            f'style="width:{p:.2f}%;background:{cm["color"]};"></div></div></div>'
        )
    st.markdown(
        f"""
        <div class="result">
          <div class="rsphere" style="background:{m['sphere']};"></div>
          <h3 class="rname" style="color:{m['color']};">{m['label']} ({m['en']})</h3>
          <span class="rconf">Confidence: {conf:.2f}%</span>
          <div class="rfield"><div class="lab">ℹ️ Deskripsi</div><div class="val">{m['deskripsi']}</div></div>
          <div class="rfield"><div class="lab">💡 Rekomendasi</div><div class="val">{m['rekomendasi']}</div></div>
          <div class="rfield"><div class="lab">📦 Penyimpanan</div><div class="val">{m['penyimpanan']}</div></div>
          <div class="rprob"><div class="lab">📊 Probabilitas Semua Kelas</div>{prob_rows}</div>
          <div class="disclaimer">Catatan akademis: hasil bergantung pada kualitas citra
            (pencahayaan, latar, sudut, fokus). Model selalu memilih salah satu dari empat
            kelas meskipun objek pada citra bukan pisang.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        f"""
        <div class="footer">
          <div class="footer-grid">
            <div>
              <div class="footer-brand"><div class="nav-logo">🍌</div><b>Banana AI</b></div>
              <p>Sistem klasifikasi kematangan pisang berbasis AI untuk membantu industri
                pertanian dan distribusi buah.</p>
            </div>
            <div>
              <h5>Quick Links</h5>
              <a href="#home">Home</a><a href="#about">About</a>
              <a href="#technology">Technology</a><a href="#features">Features</a>
            </div>
            <div>
              <h5>Technology</h5>
              <div class="fcontact"><span class="ftick">✓</span> TensorFlow / Keras</div>
              <div class="fcontact"><span class="ftick">✓</span> Streamlit</div>
              <div class="fcontact"><span class="ftick">✓</span> Deep Learning</div>
              <div class="fcontact"><span class="ftick">✓</span> MobileNetV2 (CNN)</div>
            </div>
            <div>
              <h5>Penulis</h5>
              <div class="fcontact">👤 {STUDENT_NAME}</div>
              <div class="fcontact">🆔 {STUDENT_NPM} · {STUDENT_CLASS}</div>
              <div class="fcontact">🎓 {PROGRAM_STUDI}</div>
              <div class="fcontact">🏫 {UNIVERSITAS}</div>
            </div>
          </div>
          <div class="footer-bottom">
            © 2026 <b>Banana AI</b> — Penulisan Ilmiah oleh {STUDENT_NAME} ({STUDENT_NPM}).
            Semua hak cipta dilindungi.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Bagian Get Started (widget native Streamlit di dalam wadah bergaya kartu)     #
# --------------------------------------------------------------------------- #
def render_get_started(model, idx_to_label) -> None:
    st.markdown(
        """
        <section id="get-started" class="section try-band" style="padding-bottom:40px;">
          <div class="eyebrow">Try It Now</div>
          <h2 class="h2">Coba <span class="g">Banana AI</span></h2>
          <p class="lead">Unggah gambar pisang Anda dan biarkan AI menganalisis tingkat
            kematangannya.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    # Satu kartu (keyed container) yang membungkus header + uploader + tombol + hasil.
    # Memakai st.container(key=...) supaya widget native benar-benar berada DI DALAM
    # kartu (trik <div> lintas st.markdown tidak bisa membungkus widget Streamlit).
    with st.container(key="uploadcard"):
        st.markdown(
            '<h3 class="uploadcard-title">Upload Gambar Pisang</h3>',
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "Klik atau seret gambar (PNG, JPG, JPEG · maks 16MB)",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        image = None
        file_bytes = None
        if uploaded is not None:
            file_bytes = uploaded.getvalue()
            try:
                image = _open_image(io.BytesIO(file_bytes))
                st.image(image, use_container_width=True)
            except InvalidImageError:
                st.error("File gambar tidak valid. Gunakan format JPG, JPEG, atau PNG.")
                image = None

        analyze = st.button(
            "Analisis Kematangan Sekarang",
            disabled=(image is None),
            use_container_width=True,
        )
        result_slot = st.empty()

        if image is None:
            result_slot.markdown(
                '<p style="text-align:center;color:var(--muted);margin-top:18px;">'
                'Belum ada gambar — unggah foto pisang untuk memulai analisis.</p>',
                unsafe_allow_html=True,
            )

        if analyze and image is not None and file_bytes is not None:
            # Loader pisang berputar selama prediksi berjalan
            result_slot.markdown(
                '<div class="banana-loader"><div class="banana-spin">🍌</div>'
                '<div class="txt">Menganalisis kematangan…</div></div>',
                unsafe_allow_html=True,
            )
            try:
                top_key, conf, prob_map = classify_bytes(file_bytes, model, idx_to_label)
            except Exception as exc:
                result_slot.error(f"Terjadi kesalahan saat prediksi: {exc}")
                return
            result_slot.empty()
            with result_slot.container():
                render_result_card(top_key, conf, prob_map)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    inject_css()
    metrics = load_report_metrics()

    render_navbar()
    render_hero(metrics)
    render_about(metrics)
    render_technology()
    render_features(metrics)

    # Muat model; bila gagal tampilkan pesan rapi tapi halaman tetap utuh.
    model = idx_to_label = None
    try:
        model, idx_to_label = get_artifacts()
    except Exception as exc:
        st.markdown(
            f"""
            <section id="get-started" class="section try-band">
              <div class="eyebrow">Try It Now</div>
              <h2 class="h2">Coba <span class="g">Banana AI</span></h2>
              <div class="upload-card"><h3>⚠️ Model belum tersedia</h3>
                <p style="text-align:center;color:var(--ink-soft);">{exc}</p>
                <p style="text-align:center;color:var(--muted);font-size:.9rem;">
                  Jalankan notebook pelatihan agar berkas
                  <code>models/klasifikasi_kematangan_pisang_model.keras</code> dan
                  <code>models/class_indices.json</code> terbentuk, lalu muat ulang halaman.</p>
              </div>
            </section>
            """,
            unsafe_allow_html=True,
        )

    if model is not None:
        render_get_started(model, idx_to_label)

    render_footer()


if __name__ == "__main__":
    main()
