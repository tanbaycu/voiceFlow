import os
import uuid
import tempfile
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

from transcribe_translate import (
    transcribe_audio_to_english,
    translate_en_to_vi,
)

# ==== DB setup (inline) ====
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, func
from sqlalchemy.orm import sessionmaker, DeclarativeBase


def _normalize_database_url(url: str) -> str:
    if not url:
        return "sqlite:///local.db"
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
    parsed = urlparse(url)
    qs = dict(parse_qsl(parsed.query))
    qs.setdefault("sslmode", "require")
    new_query = urlencode(qs)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment))

_DEFAULT_RAILWAY_URL = (
    "postgresql://postgres:tOhciSCnBMStiyVEyIgQAICddQrzJjDc@caboose.proxy.rlwy.net:39272/railway"
)
DATABASE_URL = _normalize_database_url(os.environ.get("DATABASE_URL") or _DEFAULT_RAILWAY_URL)

class Base(DeclarativeBase):
    pass

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

class Transcript(Base):
    __tablename__ = "transcripts"

    id = Column(Integer, primary_key=True, index=True)
    audio_filename = Column(String(255), nullable=False)
    en_text = Column(Text, nullable=False)
    vi_text = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


def get_session():
    return SessionLocal()

# ===========================


ALLOWED_EXTENSIONS = {"mp3", "wav", "m4a", "mp4", "aac", "flac", "ogg"}

# Cấu hình nhẹ cho gói Free
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "tiny")
MAX_CONTENT_LENGTH_MB = int(os.environ.get("MAX_CONTENT_LENGTH_MB", "30"))

# Tạo bảng nếu chưa có
Base.metadata.create_all(bind=engine)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH_MB * 1024 * 1024


@app.get("/")
def upload_form():
    return render_template("upload.html")


@app.post("/upload")
def handle_upload():
    if "audio" not in request.files:
        flash("Không tìm thấy file tải lên.")
        return redirect(url_for("upload_form"))

    file = request.files["audio"]
    if file.filename == "":
        flash("Chưa chọn file.")
        return redirect(url_for("upload_form"))

    if not allowed_file(file.filename):
        flash("Định dạng không được hỗ trợ.")
        return redirect(url_for("upload_form"))

    filename = secure_filename(file.filename)
    suffix = "." + filename.rsplit(".", 1)[1].lower()

    try:
        with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as tmp:
            file.save(tmp.name)
            en_text = transcribe_audio_to_english(
                tmp.name,
                model_size=WHISPER_MODEL,
                beam_size=1,
            )
    except Exception as exc:
        flash(f"Lỗi nhận dạng: {exc}")
        return redirect(url_for("upload_form"))

    return render_template(
        "preview.html",
        en_text=en_text,
        vi_text=None,
    )


@app.post("/translate")
def do_translate():
    en_text = request.form.get("en_text", "").strip()
    if not en_text:
        flash("Thiếu dữ liệu dịch.")
        return redirect(url_for("upload_form"))

    try:
        vi_text = translate_en_to_vi(en_text)
    except Exception as exc:
        flash(f"Lỗi dịch: {exc}")
        return redirect(url_for("upload_form"))

    return render_template(
        "preview.html",
        en_text=en_text,
        vi_text=vi_text,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False, use_reloader=False)
