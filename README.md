# Audio Transcription & Translation App

A Flask-based web application that transcribes English audio to text using Whisper AI and translates it to Vietnamese using Google Translator.

## Features

- 🎙️ Audio transcription (English) using Whisper AI
- 🌐 Translation from English to Vietnamese
- 📁 Support for multiple audio formats (mp3, wav, m4a, mp4, aac, flac, ogg)
- 💾 Download transcription and translation as text files
- 🎨 Modern, responsive web interface
- ⚡ Fast processing with GPU support (if available)

## Installation

1. Install dependencies:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

2. Create required directories (auto-created on first run):
\`\`\`
uploads/  # Uploaded audio files
outputs/  # Generated text files
templates/  # HTML templates
\`\`\`

## Usage

### Web Server Mode

Start the web application:
\`\`\`bash
python app.py --web
\`\`\`

Or simply:
\`\`\`bash
python app.py
\`\`\`

Then open your browser to `http://127.0.0.1:5000`

### CLI Mode

Process audio files directly from command line:
\`\`\`bash
python app.py input.mp3
\`\`\`

With custom options:
\`\`\`bash
python app.py input.mp3 --model medium --device cuda --out-en english.txt --out-vi vietnamese.txt
\`\`\`

## Configuration

Environment variables:
- `WHISPER_MODEL`: Model size (tiny, base, small, medium, large) - default: small
- `MAX_CONTENT_LENGTH_MB`: Max upload size in MB - default: 100
- `FLASK_SECRET_KEY`: Flask secret key - default: dev-secret-key
- `PORT`: Server port - default: 5000
- `HOST`: Server host - default: 127.0.0.1
- `DEBUG`: Debug mode - default: False

## API Endpoints

- `GET /` - Main web interface
- `POST /upload` - Upload and transcribe audio
- `POST /translate` - Translate English text to Vietnamese
- `GET /uploads/<filename>` - Download uploaded audio
- `GET /outputs/<filename>` - Download output text files

## Requirements

- Python 3.10+
- Flask 3.0+
- faster-whisper 1.0+
- deep-translator 1.11+
- torch 2.0+ (for GPU acceleration)

## License

MIT License
