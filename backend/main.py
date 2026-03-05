"""
FastAPI Backend — Multilingual AI Assistant
============================================
Provides REST API endpoints for:
  - Language detection   → POST /detect-language
  - Translation          → POST /translate
  - Sentiment analysis   → POST /sentiment
  - AI chat response     → POST /chat
  - Health check         → GET  /health

Models are loaded lazily on first request to each endpoint.
All model instances are module-level singletons to prevent
repeated loading across requests.

Run with:
    uvicorn backend.main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

# Import singleton module instances
from backend.language_detector import language_detector
from backend.translator import translator, LANGUAGE_CODE_MAP
from backend.sentiment import sentiment_analyzer
from backend.chatbot import chatbot


# ---------------------------------------------------------------------------
# FastAPI application setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Multilingual AI Assistant API",
    description=(
        "A production-ready multilingual API that provides language detection, "
        "translation, sentiment analysis, and AI chat using Hugging Face models."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS so the Streamlit frontend can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # In production, restrict this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request & Response Schemas (Pydantic models)
# ---------------------------------------------------------------------------

class DetectLanguageRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to detect language for")

class DetectLanguageResponse(BaseModel):
    language: str
    confidence: float


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to translate")
    source_language: str = Field(..., description="Source language (name or NLLB code)")
    target_language: str = Field(..., description="Target language (name or NLLB code)")

class TranslateResponse(BaseModel):
    translated_text: str
    source_language: str
    target_language: str


class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to analyze")

class SentimentResponse(BaseModel):
    label: str
    confidence: float
    emoji: str


class ChatRequest(BaseModel):
    text: str = Field(..., min_length=1, description="User's message")
    language: Optional[str] = Field("en", description="Language code of the input (informational)")

class ChatResponse(BaseModel):
    response: str
    model: str


class HealthResponse(BaseModel):
    status: str
    version: str


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Check that the API is running."""
    return {"status": "ok", "version": "1.0.0"}


@app.get("/languages", tags=["Utilities"])
def list_supported_languages():
    """Return the list of supported languages for translation."""
    return {
        "supported_languages": list(LANGUAGE_CODE_MAP.keys()),
        "total": len(LANGUAGE_CODE_MAP),
    }


@app.post("/detect-language", response_model=DetectLanguageResponse, tags=["Language Detection"])
def detect_language(request: DetectLanguageRequest):
    """
    Detect the language of the input text.

    Uses the papluca/xlm-roberta-base-language-detection model.
    Returns the ISO 639-1 language code and confidence score.
    """
    try:
        result = language_detector.detect(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Language detection failed: {str(e)}")


@app.post("/translate", response_model=TranslateResponse, tags=["Translation"])
def translate(request: TranslateRequest):
    """
    Translate text between languages.

    Uses the facebook/nllb-200-distilled-600M model.
    Accepts both user-friendly language names (e.g., 'French') and
    NLLB language codes (e.g., 'fra_Latn').
    """
    try:
        result = translator.translate(
            text=request.text,
            source_language=request.source_language,
            target_language=request.target_language,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")


@app.post("/sentiment", response_model=SentimentResponse, tags=["Sentiment Analysis"])
def analyze_sentiment(request: SentimentRequest):
    """
    Analyze the sentiment of the input text.

    Uses the cardiffnlp/twitter-xlm-roberta-base-sentiment model.
    Returns Positive, Negative, or Neutral with a confidence score.
    """
    try:
        result = sentiment_analyzer.analyze(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {str(e)}")


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(request: ChatRequest):
    """
    Generate an AI conversational response to the input text.

    Uses the facebook/blenderbot-400M-distill model.
    For best results, provide input in English (or pre-translate using /translate).
    """
    try:
        result = chatbot.respond(text=request.text, language=request.language)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat response generation failed: {str(e)}")
