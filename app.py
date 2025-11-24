"""
Copyright (c) 2025 tanbaycu
This file is part of projects under tanbaycu, and is released for study and
educational purposes only.
"""
"""
Audio Transcription & Translation Web Application
Combines Flask web server with Whisper transcription and Google translation
"""

import argparse
import os
import sys
import time
import uuid
import json
from typing import List, Dict, Tuple
from datetime import datetime

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from user_agents import parse

# ============================================================================
# Configuration
# ============================================================================

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
SESSIONS_DIR = os.path.join(os.path.dirname(__file__), "sessions")
ALLOWED_EXTENSIONS = {"mp3", "wav", "m4a", "mp4", "aac", "flac", "ogg"}
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SENDER_EMAIL = os.getenv("SENDER_EMAIL", "")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD", "")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SESSIONS_DIR, exist_ok=True)

WHISPER_MODEL = "small"
MAX_CONTENT_LENGTH_MB = 100


# ============================================================================
# Session Management Functions
# ============================================================================

def create_session(audio_filename: str, transcription: str, en_txt_path: str) -> str:
    """Create a new session and save to file"""
    session_id = uuid.uuid4().hex
    session_data = {
        "session_id": session_id,
        "created_at": datetime.now().isoformat(),
        "audio_filename": audio_filename,
        "transcription": transcription,
        "en_txt_path": en_txt_path,
        "translation": None,
        "vi_txt_path": None
    }
    
    session_file = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    with open(session_file, "w", encoding="utf-8") as f:
        json.dump(session_data, f, ensure_ascii=False, indent=2)
    
    return session_id


def get_session(session_id: str) -> Dict | None:
    """Retrieve session data from file"""
    session_file = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if not os.path.exists(session_file):
        return None
    
    with open(session_file, "r", encoding="utf-8") as f:
        return json.load(f)


def update_session(session_id: str, updates: Dict) -> bool:
    """Update session data in file"""
    session_data = get_session(session_id)
    if not session_data:
        return False
    
    session_data.update(updates)
    session_data["updated_at"] = datetime.now().isoformat()
    
    session_file = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    with open(session_file, "w", encoding="utf-8") as f:
        json.dump(session_data, f, ensure_ascii=False, indent=2)
    
    return True


# ============================================================================
# Core Transcription & Translation Functions
# ============================================================================

_model_cache: Dict[Tuple[str, str, str], object] = {}


def _ensure_dependencies() -> None:
    """Check if required dependencies are installed"""
    try:
        import faster_whisper  # noqa: F401
    except Exception:
        sys.stderr.write(
            "[ERROR] Missing 'faster-whisper' library. Install with: pip install faster-whisper\n"
        )
        raise
    try:
        import deep_translator  # noqa: F401
    except Exception:
        sys.stderr.write(
            "[ERROR] Missing 'deep-translator' library. Install with: pip install deep-translator\n"
        )
        raise


def _get_model(model_size: str, device: str, compute_type: str):
    """Get or create cached Whisper model"""
    from faster_whisper import WhisperModel

    key = (model_size, device, compute_type)
    model = _model_cache.get(key)
    if model is not None:
        return model

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    _model_cache[key] = model
    return model


def transcribe_audio_to_english(
    audio_path: str,
    model_size: str = "small",
    device: str = "auto",
    compute_type: str = "auto",
    beam_size: int = 5,
    vad_filter: bool = True,
) -> str:
    """Transcribe audio file to English text using Whisper"""
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"

    model = _get_model(model_size, device, compute_type)

    segments, info = model.transcribe(
        audio_path,
        language="en",
        task="transcribe",
        beam_size=beam_size,
        vad_filter=vad_filter,
        no_speech_threshold=0.6,
        condition_on_previous_text=True,
    )

    text_parts: List[str] = []
    for seg in segments:
        chunk = seg.text.strip()
        if chunk:
            text_parts.append(chunk)

    return normalize_spaces(" ".join(text_parts).strip())


def normalize_spaces(s: str) -> str:
    """Normalize whitespace in text"""
    return " ".join(s.split())


def _split_into_chunks(text: str, max_len: int = 4500) -> List[str]:
    """Split text into chunks for translation"""
    if len(text) <= max_len:
        return [text]

    import re
    sentences = re.split(r"(?<=[\.!?])\s+", text)
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for sent in sentences:
        if not sent:
            continue
        if current_len + len(sent) + 1 <= max_len:
            current.append(sent)
            current_len += len(sent) + 1
        else:
            chunks.append(" ".join(current))
            current = [sent]
            current_len = len(sent)

    if current:
        chunks.append(" ".join(current))

    return chunks


def translate_en_to_vi(text: str, retries: int = 3, cooldown_sec: float = 1.0) -> str:
    """Translate English text to Vietnamese using Google Translator"""
    from deep_translator import GoogleTranslator

    chunks = _split_into_chunks(text)
    vi_parts: List[str] = []

    translator = GoogleTranslator(source="en", target="vi")

    for chunk in chunks:
        attempt = 0
        while True:
            try:
                vi = translator.translate(chunk)
                vi_parts.append(vi)
                break
            except Exception as exc:
                attempt += 1
                if attempt > retries:
                    raise RuntimeError(f"Translation failed after {retries} attempts: {exc}")
                time.sleep(cooldown_sec)

    return "\n\n".join(vi_parts).strip()


def write_text_file(path: str, content: str) -> None:
    """Write text content to file"""
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def read_text_file(path: str) -> str:
    """Read text content from file"""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def derive_paths(input_audio: str, out_en: str | None, out_vi: str | None) -> tuple[str, str]:
    """Derive output file paths from input audio path"""
    base, _ = os.path.splitext(input_audio)
    en_path = out_en or f"{base}.en.txt"
    vi_path = out_vi or f"{base}.vi.txt"
    return en_path, vi_path

# ============================================================================
# Email Functions
# ============================================================================

def send_email(to_email: str, subject: str, body: str) -> bool:
    """Send email with feedback notification"""
    try:
        if not SENDER_EMAIL or not SENDER_PASSWORD:
            print("[WARNING] Email not configured. Skipping email send.")
            return False
        
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = to_email
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'html'))
        
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
        
        return True
    except Exception as e:
        print(f"[ERROR] Failed to send email: {e}")
        return False

# ============================================================================
# Flask Application
# ============================================================================

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# Ensure dependencies on startup
_ensure_dependencies()

app = Flask(__name__)
app.secret_key = "voiceflow-secret-key-2024"
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH_MB * 1024 * 1024


@app.get("/")
def index():
    """Serve the main HTML page - desktop or mobile version based on user agent"""
    user_agent = request.headers.get('User-Agent', '')
    ua = parse(user_agent)
    is_mobile = ua.is_mobile or ua.is_tablet
    
    if is_mobile:
        return render_template("mobile.html")
    return render_template("index.html")


@app.get("/<session_id>")
def session_view(session_id: str):
    """Serve the main HTML page with session context - desktop or mobile version"""
    session_data = get_session(session_id)
    
    user_agent = request.headers.get('User-Agent', '')
    ua = parse(user_agent)
    is_mobile = ua.is_mobile or ua.is_tablet
    
    if not session_data:
        return render_template("mobile.html" if is_mobile else "index.html")
    
    return render_template("mobile.html" if is_mobile else "index.html", session_id=session_id)


@app.get("/api/session/<session_id>")
def get_session_api(session_id: str):
    """API endpoint to retrieve session data"""
    session_data = get_session(session_id)
    if not session_data:
        return jsonify({"success": False, "error": "Session not found"}), 404
    
    return jsonify({
        "success": True,
        "session": session_data
    })


@app.post("/upload")
def handle_upload():
    """Handle audio file upload and transcription"""
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    file = request.files["audio"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "File format not supported"}), 400

    filename = secure_filename(file.filename)
    unique_prefix = uuid.uuid4().hex[:8]
    saved_name = f"{unique_prefix}_{filename}"
    saved_path = os.path.join(UPLOAD_DIR, saved_name)
    file.save(saved_path)

    try:
        en_text = transcribe_audio_to_english(
            saved_path,
            model_size=WHISPER_MODEL,
            beam_size=5,
        )
    except Exception as exc:
        return jsonify({"error": f"Transcription error: {str(exc)}"}), 500

    en_txt_path, _ = derive_paths(saved_path, None, None)
    en_txt_path = os.path.join(OUTPUT_DIR, os.path.basename(en_txt_path))
    write_text_file(en_txt_path, en_text)

    session_id = create_session(saved_name, en_text, en_txt_path)

    return jsonify({
        "success": True,
        "session_id": session_id,
        "session_url": f"/{session_id}",
        "audio_url": f"/uploads/{saved_name}",
        "transcription": en_text,
        "en_txt_url": f"/outputs/{os.path.basename(en_txt_path)}",
        "audio_filename": saved_name
    })


@app.post("/retry")
def handle_retry():
    """Handle retry request using existing session"""
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    session_id = data.get("session_id", "").strip()
    
    if not session_id:
        return jsonify({"error": "No session ID provided"}), 400
    
    session_data = get_session(session_id)
    if not session_data:
        return jsonify({"error": "Session not found"}), 404
    
    audio_filename = session_data["audio_filename"]
    saved_path = os.path.join(UPLOAD_DIR, audio_filename)
    
    if not os.path.exists(saved_path):
        return jsonify({"error": "Audio file not found"}), 404
    
    try:
        en_text = transcribe_audio_to_english(
            saved_path,
            model_size=WHISPER_MODEL,
            beam_size=5,
        )
    except Exception as exc:
        return jsonify({"error": f"Transcription error: {str(exc)}"}), 500
    
    en_txt_path, _ = derive_paths(saved_path, None, None)
    en_txt_path = os.path.join(OUTPUT_DIR, os.path.basename(en_txt_path))
    write_text_file(en_txt_path, en_text)
    
    update_session(session_id, {
        "transcription": en_text,
        "en_txt_path": en_txt_path
    })
    
    return jsonify({
        "success": True,
        "audio_url": f"/uploads/{audio_filename}",
        "transcription": en_text,
        "en_txt_url": f"/outputs/{os.path.basename(en_txt_path)}",
        "audio_filename": audio_filename
    })


@app.post("/translate")
def do_translate():
    """Handle translation from English to Vietnamese"""
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    session_id = data.get("session_id", "").strip()
    en_text = data.get("en_text", "").strip()
    audio_filename = data.get("audio_filename", "").strip()
    
    if not en_text or not audio_filename:
        return jsonify({"error": "Missing required data"}), 400

    try:
        vi_text = translate_en_to_vi(en_text)
    except Exception as exc:
        return jsonify({"error": f"Translation error: {str(exc)}"}), 500

    saved_path = os.path.join(UPLOAD_DIR, audio_filename)
    _, vi_default = derive_paths(saved_path, None, None)
    vi_txt_path = os.path.join(OUTPUT_DIR, os.path.basename(vi_default))
    write_text_file(vi_txt_path, vi_text)

    if session_id:
        update_session(session_id, {
            "translation": vi_text,
            "vi_txt_path": vi_txt_path
        })

    return jsonify({
        "success": True,
        "translation": vi_text,
        "vi_txt_url": f"/outputs/{os.path.basename(vi_txt_path)}"
    })


@app.get("/uploads/<path:filename>")
def download_uploaded(filename: str):
    """Serve uploaded audio files"""
    path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
    return send_file(path, as_attachment=True)


@app.get("/outputs/<path:filename>")
def download_output(filename: str):
    """Serve output text files"""
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
    return send_file(path, as_attachment=True)

@app.get("/dm")
def dm_page():
    """Serve the feedback/DM page"""
    return render_template("dm.html")


@app.post("/api/feedback")
def handle_feedback():
    """Handle feedback submission with dual email notifications"""
    data = request.get_json()
    
    if not data:
        return jsonify({"success": False, "error": "No data provided"}), 400
    
    email = data.get("email", "").strip()
    subject = data.get("subject", "").strip() or "No Subject"
    message = data.get("message", "").strip()
    
    if not email or not message:
        return jsonify({"success": False, "error": "Email and message are required"}), 400
    
    # Validate email format
    if "@" not in email or "." not in email:
        return jsonify({"success": False, "error": "Invalid email format"}), 400
    
    user_subject = "Cảm ơn bạn đã gửi phản hồi - VoiceFlow"
    user_body = f"""
    <html>
        <body style="font-family: 'Segoe UI', Arial, sans-serif; color: #333; line-height: 1.6;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                <h2 style="color: #667eea;">Cảm ơn bạn! 🎉</h2>
                <p>Tôi đã nhận được phản hồi của bạn và rất trân trọng ý kiến của bạn.</p>
                
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1px; border-radius: 8px; margin: 20px 0;">
                    <div style="background: white; padding: 15px; border-radius: 7px;">
                        <p><strong style="color: #667eea;">Tiêu đề:</strong> {subject}</p>
                        <p><strong style="color: #667eea;">Nội dung:</strong></p>
                        <p style="white-space: pre-wrap; color: #555;">{message}</p>
                    </div>
                </div>
                
                <p>Tôi sẽ sớm xem xét và liên hệ với bạn nếu cần thêm thông tin.</p>
                
                <hr style="border: none; border-top: 1px solid #eee; margin: 30px 0;">
                <p style="font-size: 12px; color: #999; text-align: center;">
                    VoiceFlow - Voice to Text Conversion Tool<br>
                    Đây là email tự động. Vui lòng không trả lời email này.
                </p>
            </div>
        </body>
    </html>
    """
    
    user_email_sent = send_email(email, user_subject, user_body)
    
    admin_email = "tranminhtan4953@gmail.com"
    admin_subject = f"🔔 Phản hồi từ {email} - VoiceFlow"
    admin_body = f"""
    <html>
        <body style="font-family: 'Segoe UI', Arial, sans-serif; color: #333; line-height: 1.6;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                <h2 style="color: #667eea;">📬 Phản hồi mới từ người dùng</h2>
                
                <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 20px 0;">
                    <p><strong>Email:</strong> <a href="mailto:{email}" style="color: #667eea;">{email}</a></p>
                    <p><strong>Tiêu đề:</strong> {subject}</p>
                    <p><strong>Thời gian:</strong> {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
                </div>
                
                <h3 style="color: #667eea; margin-top: 25px;">Nội dung phản hồi:</h3>
                <div style="background: #fff; border-left: 4px solid #667eea; padding: 15px; margin: 15px 0; border-radius: 4px;">
                    <p style="white-space: pre-wrap; margin: 0; color: #555;">{message}</p>
                </div>
                
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; border-radius: 8px; margin: 20px 0; text-align: center;">
                    <p style="margin: 0;"><strong>Email cảm ơn đã được gửi tới người dùng: {('✅' if user_email_sent else '❌')}</strong></p>
                </div>
                
                <p style="margin-top: 30px;">
                    <a href="mailto:{email}?subject=Re: {subject.replace(' ', '%20')}" style="display: inline-block; background: #667eea; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">
                        Trả lời
                    </a>
                </p>
                
                <hr style="border: none; border-top: 1px solid #eee; margin: 30px 0;">
                <p style="font-size: 11px; color: #999; text-align: center;">
                    VoiceFlow Admin Dashboard<br>
                    Đây là thông báo tự động từ hệ thống.
                </p>
            </div>
        </body>
    </html>
    """
    
    admin_email_sent = send_email(admin_email, admin_subject, admin_body)
    
    return jsonify({
        "success": True, 
        "message": "Feedback sent successfully",
        "user_email_sent": user_email_sent,
        "admin_email_sent": admin_email_sent
    }), 200


# ============================================================================
# CLI Interface (Optional)
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments for CLI mode"""
    p = argparse.ArgumentParser(
        description="Transcribe audio (English) to text and translate to Vietnamese"
    )
    p.add_argument("input", nargs="?", help="Audio file path (mp3/wav/m4a ...)")
    p.add_argument("--out-en", dest="out_en", default=None, help="English text output path")
    p.add_argument("--out-vi", dest="out_vi", default=None, help="Vietnamese text output path")
    p.add_argument("--model", dest="model", default="small", help="Whisper model size (tiny, base, small, medium, large)")
    p.add_argument("--device", dest="device", default="auto", help="cpu | cuda | auto")
    p.add_argument("--compute-type", dest="compute_type", default="auto", help="float16 | int8 | auto")
    p.add_argument("--no-vad", dest="no_vad", action="store_true", help="Disable VAD filter")
    p.add_argument("--web", dest="web", action="store_true", help="Run web server mode")
    return p.parse_args()


def cli_main() -> None:
    """CLI mode for direct file processing"""
    args = parse_args()

    audio_path = args.input
    if not os.path.isfile(audio_path):
        sys.stderr.write(f"[ERROR] File not found: {audio_path}\n")
        sys.exit(1)

    out_en, out_vi = derive_paths(audio_path, args.out_en, args.out_vi)

    try:
        en_text = transcribe_audio_to_english(
            audio_path=audio_path,
            model_size=args.model,
            device=args.device,
            compute_type=args.compute_type,
            vad_filter=not args.no_vad,
        )
    except Exception as exc:
        sys.stderr.write(f"[ERROR] Transcription error: {exc}\n")
        sys.exit(2)

    try:
        write_text_file(out_en, en_text)
    except Exception as exc:
        sys.stderr.write(f"[ERROR] Cannot write English file: {exc}\n")
        sys.exit(3)

    try:
        vi_text = translate_en_to_vi(en_text)
    except Exception as exc:
        sys.stderr.write(f"[ERROR] Translation error: {exc}\n")
        sys.exit(4)

    try:
        write_text_file(out_vi, vi_text)
    except Exception as exc:
        sys.stderr.write(f"[ERROR] Cannot write Vietnamese file: {exc}\n")
        sys.exit(5)

    print("Completed successfully!")
    print(f"- English text: {out_en}")
    print(f"- Vietnamese translation: {out_vi}")


if __name__ == "__main__":
    args = parse_args()
    
    if args.web or not args.input:
        port = 5000
        host = "0.0.0.0"  
        debug = False
        
        print(f"Starting web server on {host}:{port}")
        app.run(host=host, port=port, debug=debug, use_reloader=False)
    else:
        # CLI mode
        cli_main()
