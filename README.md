## Transcribe & Translate (EN -> VI)

Script chuyển file âm thanh tiếng Anh (mp3/wav/m4a, ...) thành văn bản .txt và dịch sang tiếng Việt.

### Cài đặt

1) Cài Python 3.10+.
2) Mở PowerShell tại thư mục dự án và chạy:
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Lưu ý: Nếu cần tăng tốc GPU trên Windows, cài torch phù hợp theo hướng dẫn của PyTorch.

### Sử dụng

```bash
python transcribe_translate.py <duong_dan_file_audio> [--out-en EN_TXT] [--out-vi VI_TXT] [--model MODEL]
```

- `--model`: tiny | base | small | medium | large (mặc định: small)
- Tự động chọn `cuda` nếu có, ngược lại `cpu`.

Ví dụ:
```bash
python transcribe_translate.py 2025_3_10_27_15_913_9.mp3 --model small
```

Kết quả:
- Tạo `<ten_file>.en.txt`: bản tiếng Anh
- Tạo `<ten_file>.vi.txt`: bản dịch tiếng Việt
