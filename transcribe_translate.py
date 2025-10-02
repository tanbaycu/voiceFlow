import argparse
import os
import sys
import time
from typing import Iterable, List


def _ensure_dependencies() -> None:
    try:
        import faster_whisper  # noqa: F401
    except Exception as exc:  # pragma: no cover
        sys.stderr.write(
            "[ERROR] Thiếu thư viện 'faster-whisper'. Hãy cài đặt bằng: pip install -r requirements.txt\n"
        )
        raise
    try:
        import deep_translator  # noqa: F401
    except Exception:  # pragma: no cover
        sys.stderr.write(
            "[ERROR] Thiếu thư viện 'deep-translator'. Hãy cài đặt bằng: pip install -r requirements.txt\n"
        )
        raise


def transcribe_audio_to_english(
    audio_path: str,
    model_size: str = "small",
    device: str = "auto",
    compute_type: str = "auto",
    beam_size: int = 5,
    vad_filter: bool = True,
) -> str:
    from faster_whisper import WhisperModel

    if device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"

    model = WhisperModel(model_size, device=device, compute_type=compute_type)

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
    for seg in segments:  # type: ignore[assignment]
        chunk = seg.text.strip()
        if chunk:
            text_parts.append(chunk)

    return normalize_spaces(" ".join(text_parts).strip())


def normalize_spaces(s: str) -> str:
    return " ".join(s.split())


def write_text_file(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _split_into_chunks(text: str, max_len: int = 4500) -> List[str]:
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
            except Exception as exc:  # pragma: no cover
                attempt += 1
                if attempt > retries:
                    raise RuntimeError(f"Dich that bai sau {retries} lan: {exc}")
                time.sleep(cooldown_sec)

    return "\n\n".join(vi_parts).strip()


def derive_paths(input_audio: str, out_en: str | None, out_vi: str | None) -> tuple[str, str]:
    base, _ = os.path.splitext(input_audio)
    en_path = out_en or f"{base}.en.txt"
    vi_path = out_vi or f"{base}.vi.txt"
    return en_path, vi_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Transcribe MP3 (tiếng Anh) sang văn bản .txt và dịch sang tiếng Việt"
        )
    )
    p.add_argument("input", help="Đường dẫn file âm thanh (mp3/wav/m4a ...)")
    p.add_argument("--out-en", dest="out_en", default=None, help="Đường dẫn .txt tiếng Anh")
    p.add_argument("--out-vi", dest="out_vi", default=None, help="Đường dẫn .txt tiếng Việt")
    p.add_argument("--model", dest="model", default="small", help="Kích thước model Whisper (tiny, base, small, medium, large)")
    p.add_argument("--device", dest="device", default="auto", help="cpu | cuda | auto")
    p.add_argument("--compute-type", dest="compute_type", default="auto", help="float16 | int8 | auto")
    p.add_argument("--no-vad", dest="no_vad", action="store_true", help="Tắt VAD filter")
    return p.parse_args()


def main() -> None:
    _ensure_dependencies()
    args = parse_args()

    audio_path = args.input
    if not os.path.isfile(audio_path):
        sys.stderr.write(f"[ERROR] Không tìm thấy file: {audio_path}\n")
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
        sys.stderr.write(f"[ERROR] Lỗi khi nhận dạng giọng nói: {exc}\n")
        sys.exit(2)

    try:
        write_text_file(out_en, en_text)
    except Exception as exc:
        sys.stderr.write(f"[ERROR] Không ghi được file tiếng Anh: {exc}\n")
        sys.exit(3)

    try:
        vi_text = translate_en_to_vi(en_text)
    except Exception as exc:
        sys.stderr.write(f"[ERROR] Lỗi khi dịch sang tiếng Việt: {exc}\n")
        sys.exit(4)

    try:
        write_text_file(out_vi, vi_text)
    except Exception as exc:
        sys.stderr.write(f"[ERROR] Không ghi được file tiếng Việt: {exc}\n")
        sys.exit(5)

    print("Hoàn thành.")
    print(f"- Văn bản tiếng Anh: {out_en}")
    print(f"- Bản dịch tiếng Việt: {out_vi}")


if __name__ == "__main__":
    main()


