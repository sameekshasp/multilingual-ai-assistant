---
title: Multilingual AI Assistant
emoji: 🌍
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
---


# multilingual-ai-assistant-powered -by-huggingface-models
A multilingual AI assistant that performs **language detection**, **translation**, **sentiment analysis**, and **AI-powered chat** ,all using state-of-the-art Hugging Face Transformer models.

# 🌐 Multilingual AI Assistant

A production-ready multilingual AI assistant that performs **language detection**, **translation**, **sentiment analysis**, and **AI-powered chat** — all using state-of-the-art Hugging Face Transformer models.

---

## 🚀 Features

| Feature | Model Used |
|---------|-----------|
| 🔍 Language Detection | `papluca/xlm-roberta-base-language-detection` |
| 🌍 Translation (200+ languages) | `facebook/nllb-200-distilled-600M` |
| 💭 Sentiment Analysis | `cardiffnlp/twitter-xlm-roberta-base-sentiment` |
| 🤖 AI Chat | `facebook/blenderbot-400M-distill` |

---

## 📁 Project Structure

```
multilingual_app/
│
├── backend/
│   ├── __init__.py           # Package init
│   ├── main.py               # FastAPI app & all endpoints
│   ├── language_detector.py  # Language detection module
│   ├── translator.py         # Translation module
│   ├── sentiment.py          # Sentiment analysis module
│   └── chatbot.py            # AI chatbot module
│
├── frontend/
│   └── app.py                # Streamlit UI
│
├── requirements.txt          # All Python dependencies
└── README.md                 # This file
```

---

## 🛠️ Prerequisites

- **Python 3.10 or higher** — [Download Python](https://www.python.org/downloads/)
- **pip** (comes with Python)
- **~6 GB of disk space** for model downloads (cached after first run)
- Internet connection for first-time model downloads

---

## ⚙️ Setup & Installation

### Step 1 — Clone or navigate to the project folder

```bash
cd "c:\Users\sameeksha\Desktop\multilingual_app"
```

### Step 2 — (Recommended) Create a virtual environment

```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### Step 3 — Install all dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **Note:** If you have a CUDA-capable GPU and want GPU acceleration, install PyTorch with CUDA support first from [pytorch.org](https://pytorch.org/get-started/locally/) before running `pip install -r requirements.txt`.

---

## ▶️ Running the Application

You need **two separate terminals** — one for the backend and one for the frontend.

### Terminal 1 — Start the FastAPI Backend

```bash
cd "c:\Users\sameeksha\Desktop\multilingual_app"
uvicorn backend.main:app --reload --port 8000
```

The backend will be available at: **http://localhost:8000**

- 📄 Interactive API docs: http://localhost:8000/docs
- 📘 Alternative docs: http://localhost:8000/redoc

> 💡 **First run**: The models will download automatically from Hugging Face (~3–5 GB total). This may take several minutes. Subsequent runs will use the cached models and start much faster.

---

### Terminal 2 — Start the Streamlit Frontend

```bash
cd "c:\Users\sameeksha\Desktop\multilingual_app"
streamlit run frontend/app.py
```

The frontend will open automatically at: **http://localhost:8501**

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/languages` | List supported languages |
| `POST` | `/detect-language` | Detect language of text |
| `POST` | `/translate` | Translate text between languages |
| `POST` | `/sentiment` | Analyze sentiment of text |
| `POST` | `/chat` | Generate AI chat response |

### Example API Usage

**Detect Language:**
```bash
curl -X POST http://localhost:8000/detect-language \
  -H "Content-Type: application/json" \
  -d '{"text": "Bonjour le monde"}'

# Response: {"language": "fr", "confidence": 0.9998}
```

**Translate:**
```bash
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "source_language": "English", "target_language": "French"}'

# Response: {"translated_text": "Bonjour monde", "source_language": "eng_Latn", "target_language": "fra_Latn"}
```

**Sentiment Analysis:**
```bash
curl -X POST http://localhost:8000/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this application!"}'

# Response: {"label": "Positive", "confidence": 0.9876, "emoji": "😊"}
```

**Chat:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello! How are you?", "language": "en"}'

# Response: {"response": "I am doing well, thank you for asking!", "model": "facebook/blenderbot-400M-distill"}
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────┐
│     Streamlit Frontend          │
│  (http://localhost:8501)        │
│                                 │
│  • Text Input                   │
│  • Language Selector            │
│  • Action Buttons               │
│  • Results Display              │
└──────────────┬──────────────────┘
               │ HTTP REST API
               ▼
┌─────────────────────────────────┐
│     FastAPI Backend             │
│  (http://localhost:8000)        │
│                                 │
│  ┌──────────────────────────┐  │
│  │  language_detector.py    │  │  ← papluca/xlm-roberta-base-language-detection
│  ├──────────────────────────┤  │
│  │  translator.py           │  │  ← facebook/nllb-200-distilled-600M
│  ├──────────────────────────┤  │
│  │  sentiment.py            │  │  ← cardiffnlp/twitter-xlm-roberta-base-sentiment
│  ├──────────────────────────┤  │
│  │  chatbot.py              │  │  ← facebook/blenderbot-400M-distill
│  └──────────────────────────┘  │
└─────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Hugging Face Model Cache       │
│  (~/.cache/huggingface/)        │
└─────────────────────────────────┘
```

---

## 🌍 Supported Languages for Translation

Arabic · Bengali · Chinese (Simplified) · Chinese (Traditional) · Danish · Dutch · English · Finnish · French · German · Greek · Gujarati · Hebrew · Hindi · Indonesian · Italian · Japanese · Kannada · Korean · Malay · Marathi · Norwegian · Polish · Portuguese · Punjabi · Russian · Spanish · Swahili · Swedish · Tamil · Telugu · Thai · Turkish · Urdu · Vietnamese

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| `Connection refused` on frontend | Make sure the backend is running first |
| `Request timed out` | Model is still loading — wait 30–60 seconds and retry |
| Out of memory error | Close other apps; the models require ~4 GB RAM minimum |
| CUDA errors | Set `device_map="cpu"` in model loading, or reinstall CPU-only PyTorch |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` from project root |

---

## 📜 License

This project uses open-source Hugging Face models. See each model's license on the Hugging Face Hub.
