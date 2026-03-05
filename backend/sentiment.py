"""
Sentiment Analysis Module
=========================
Uses the cardiffnlp/twitter-xlm-roberta-base-sentiment model to perform
multilingual sentiment analysis on input text.

The model returns one of three labels:
  - Positive
  - Negative
  - Neutral

It supports 8 languages: Arabic, English, French, German,
Hindi, Italian, Portuguese, and Spanish.
"""

from transformers import pipeline


# Emoji mapping for sentiment labels
SENTIMENT_EMOJI = {
    "positive": "😊",
    "negative": "😞",
    "neutral": "😐",
}


class SentimentAnalyzer:
    """
    Analyzes the sentiment of a given text using a multilingual model.

    The model is loaded lazily on first use and cached as a singleton.
    """

    _instance = None  # Singleton instance

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
        return cls._instance

    def _load_model(self):
        """Load the sentiment analysis pipeline if not already loaded."""
        if self._model is None:
            print("[SentimentAnalyzer] Loading model: cardiffnlp/twitter-xlm-roberta-base-sentiment")
            self._model = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
                top_k=1,
            )
            print("[SentimentAnalyzer] Model loaded successfully.")

    def analyze(self, text: str) -> dict:
        """
        Analyze the sentiment of the input text.

        Args:
            text: Input text in any supported language.

        Returns:
            dict with keys:
                - label (str): Sentiment label ('Positive', 'Negative', 'Neutral')
                - confidence (float): Confidence score between 0 and 1
                - emoji (str): Emoji representing the sentiment
        """
        self._load_model()

        if not text or not text.strip():
            return {"label": "Neutral", "confidence": 0.0, "emoji": "😐"}

        # Truncate text to avoid token limit issues
        results = self._model(text[:512])
        top_result = results[0][0] if isinstance(results[0], list) else results[0]

        label = top_result["label"].lower()
        # Normalize label (model may return 'LABEL_0', etc. on some configs)
        label_map = {
            "positive": "Positive",
            "negative": "Negative",
            "neutral": "Neutral",
            "label_0": "Negative",
            "label_1": "Neutral",
            "label_2": "Positive",
        }
        normalized_label = label_map.get(label, label.capitalize())

        return {
            "label": normalized_label,
            "confidence": round(top_result["score"], 4),
            "emoji": SENTIMENT_EMOJI.get(normalized_label.lower(), "🤔"),
        }


# Module-level singleton instance
sentiment_analyzer = SentimentAnalyzer()
