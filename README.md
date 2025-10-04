# VoiceFlow

Ứng dụng web chuyển đổi giọng nói tiếng Anh thành văn bản và dịch sang tiếng Việt, sử dụng Whisper AI và Google Translator.

**Tác giả:** [tanbaycu](https://github.com/tanbaycu)
<img width="1920" height="911" alt="image" src="https://github.com/user-attachments/assets/87e89a71-55e1-4179-a052-d26681fd4436" />
<img width="1920" height="911" alt="image" src="https://github.com/user-attachments/assets/f950182a-75fa-4833-8453-fddaa03f8e0a" />


---

## Tính Năng

- Nhận dạng giọng nói tiếng Anh với độ chính xác cao (Whisper AI)
- Dịch tự động sang tiếng Việt (Google Translator)
- Hỗ trợ nhiều định dạng: MP3, WAV, M4A, MP4, AAC, FLAC, OGG (tối đa 100MB)
- Đồng bộ âm thanh với văn bản khi phát lại
- Tải xuống kết quả dưới dạng file TXT
- Quản lý phiên với URL riêng cho mỗi lần dịch
- Giao diện responsive, hỗ trợ dark mode
- Xử lý nhanh với GPU acceleration (tùy chọn)
- Retry thông minh không cần upload lại

---

## Yêu Cầu Hệ Thống

- Python 3.10+
- RAM: Tối thiểu 4GB (khuyến nghị 8GB+)
- Ổ cứng: 2GB trống cho model
- GPU NVIDIA với CUDA (tùy chọn, để tăng tốc)

---

## Cài Đặt

1. Clone repository:
```bash
git clone https://github.com/tanbaycu/voiceflow.git
cd voiceflow
```

2. Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

3. Khởi chạy ứng dụng:
```bash
python app.py --web
```

4. Truy cập: `http://127.0.0.1:5000`

---

## Sử Dụng

### Chế Độ Web

1. Chọn file âm thanh (nhấn nút hoặc kéo thả)
2. Hệ thống tự động xử lý và hiển thị kết quả
3. Nghe lại audio với tính năng highlight từ đang phát
4. Nhấn "Dịch ngay" để chuyển sang tiếng Việt
5. Copy hoặc download kết quả

### Chế Độ CLI

Xử lý file đơn giản:
```bash
python app.py input.mp3
```

Với tùy chọn nâng cao:
```bash
python app.py input.mp3 --model medium --device cuda --out-en english.txt --out-vi vietnamese.txt
```

Các tùy chọn:
- `--model`: Kích thước model (tiny, base, small, medium, large) - mặc định: small
- `--device`: Thiết bị xử lý (cpu, cuda) - mặc định: cpu
- `--out-en`: File output tiếng Anh
- `--out-vi`: File output tiếng Việt
- `--web`: Khởi chạy web server

---

## API Endpoints

### Web Interface
- `GET /` - Giao diện chính
- `GET /<session_id>` - Truy cập phiên cụ thể

### API
- `POST /upload` - Upload và nhận dạng file
- `POST /translate` - Dịch văn bản
- `POST /retry` - Thử lại xử lý
- `GET /api/session/<session_id>` - Lấy thông tin phiên
- `GET /uploads/<filename>` - Download file audio
- `GET /outputs/<filename>` - Download file văn bản

---

## Cấu Trúc Thư Mục

```bash
voiceflow/
├── app.py                 # File chính
├── requirements.txt       # Dependencies
├── README.md             # Tài liệu
├── index.html            # Hướng dẫn (GitHub Pages)
├── templates/
│   └── index.html        # Template web
├── uploads/              # File âm thanh (tự động tạo)
├── outputs/              # File văn bản (tự động tạo)
└── sessions/             # Session data (tự động tạo)
```

---

## Công Nghệ

- **Backend:** Flask 3.0+
- **AI/ML:** Whisper AI (faster-whisper), Google Translator (deep-translator)
- **Frontend:** HTML5, CSS3, JavaScript, Tailwind CSS, Lucide Icons
- **Processing:** PyTorch, NumPy

---

## Cấu Hình

Tùy chỉnh qua biến môi trường:

| Biến | Mô Tả | Mặc Định |
|------|-------|----------|
| `WHISPER_MODEL` | Kích thước model | `small` |
| `MAX_CONTENT_LENGTH_MB` | Kích thước file tối đa (MB) | `100` |
| `PORT` | Cổng server | `5000` |
| `HOST` | Địa chỉ host | `127.0.0.1` |

Ví dụ:
```bash
export WHISPER_MODEL=medium
export MAX_CONTENT_LENGTH_MB=200
python app.py --web
```

---

## Mẹo Sử Dụng
<img width="1920" height="911" alt="image" src="https://github.com/user-attachments/assets/276fb05a-1550-4326-ae0a-71ec4d69c79f" />
<img width="1920" height="911" alt="image" src="https://github.com/user-attachments/assets/37d0a3f5-2200-4af6-bcdb-2598784e1ba6" />
<img width="1920" height="911" alt="image" src="https://github.com/user-attachments/assets/f990dba7-117a-494a-bebd-632562c5f0ca" />
<img width="1920" height="911" alt="image" src="https://github.com/user-attachments/assets/936040b3-d2e4-45c3-91f3-2169fc8596a7" />


**Tăng tốc độ:**
- Sử dụng GPU với `--device cuda`
- Chọn model nhỏ hơn (tiny, base)

**Cải thiện độ chính xác:**
- Sử dụng file âm thanh chất lượng cao, ít nhiễu
- Giọng nói rõ ràng
- Sử dụng model lớn hơn (medium, large)

**Quản lý session:**
- Mỗi session có URL riêng (ví dụ: `/abc123`)
- Bookmark URL để quay lại sau
- Sử dụng "Thử lại" để xử lý lại mà không cần upload lại

---

## Bảo Mật

- Chạy hoàn toàn local, không gửi dữ liệu ra ngoài
- Không thu thập dữ liệu người dùng
- File được lưu cục bộ trên máy
- Bạn có toàn quyền quản lý và xóa dữ liệu

---


---

## Đóng Góp

Mọi đóng góp đều được hoan nghênh! Vui lòng tạo issue hoặc pull request trên GitHub.

---

## Liên Hệ

- GitHub: [@tanbaycu](https://github.com/tanbaycu)
- Issues: [GitHub Issues](https://github.com/tanbaycu/voiceflow/issues)
- Live Preview: [voiceFlow](https://tanbaycu.github.io/voiceFlow/)
---

**Phát triển bởi tanbaycu**
