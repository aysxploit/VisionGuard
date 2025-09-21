# VisionGuard – Automated License Plate Recognition (ALPR)

VisionGuard is a desktop ALPR system built with **Python 3.13**, **OpenCV**, and **Tesseract OCR**, with optional cleanup via **Google Gemini AI**.  
It supports real-time camera capture, static image recognition, SQLite logging, and a minimal Tkinter GUI.

---

## Features
- Real-time plate detection from camera feed  
- Image-based OCR recognition  
- SQLite database logging of results  
- Minimal Tkinter GUI for live video and history browsing  
- Optional Gemini API cleanup for noisy OCR output  
- Configurable via `.env` and `config.ini`  

---

## Requirements
- **Python 3.13+**  
- **Tesseract OCR** installed and added to PATH  
  - Windows: [UB Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)  
  - macOS: `brew install tesseract`  
  - Linux (Debian/Ubuntu): `sudo apt-get install tesseract-ocr`  

Install Python dependencies:
```bash
pip install -r requirements.txt
```

---

## Quick Start

Clone the repository:
```bash
git clone https://github.com/TwilightAshen3196/visionguard-fyp.git
cd visionguard-fyp
```

Set up environment:
```bash
cp .env.example .env   # Add Gemini API key if using AI cleanup
```

Run in camera mode:
```bash
python -m src.main --camera
```

Run on an image:
```bash
python -m src.main --image path/to/file.jpg
```

---

## Configuration

Settings are stored in `config.ini`:

- **[app]** → camera index, database path, snapshot folder  
- **[processing]** → thresholds, blur size, contour area, aspect ratio  
- **[gemini]** → API toggle and model selection  

---

## Project Structure
```
visionguard-fyp/
├─ data/db/         # SQLite database
├─ data/logs/       # logs + snapshots
├─ src/             # source code
├─ tests/           # pytest tests
├─ config.ini
├─ .env.example
├─ requirements.txt
└─ README.md
```

---

## License
MIT License – free to use, modify, and distribute.
