"""
Translation Module
==================
Uses the facebook/nllb-200-distilled-600M model from Hugging Face to translate
text between 200+ languages using the NLLB (No Language Left Behind) model.

Language codes follow the NLLB BCP-47 format (e.g., 'eng_Latn' for English).
A mapping from user-friendly language names to NLLB codes is provided below.
"""

from transformers import pipeline


# Mapping from user-friendly language names to NLLB-200 language codes
LANGUAGE_CODE_MAP = {
    "English": "eng_Latn",
    "French": "fra_Latn",
    "Spanish": "spa_Latn",
    "German": "deu_Latn",
    "Italian": "ita_Latn",
    "Portuguese": "por_Latn",
    "Dutch": "nld_Latn",
    "Russian": "rus_Cyrl",
    "Chinese (Simplified)": "zho_Hans",
    "Chinese (Traditional)": "zho_Hant",
    "Japanese": "jpn_Jpan",
    "Korean": "kor_Hang",
    "Arabic": "arb_Arab",
    "Hindi": "hin_Deva",
    "Bengali": "ben_Beng",
    "Urdu": "urd_Arab",
    "Turkish": "tur_Latn",
    "Vietnamese": "vie_Latn",
    "Polish": "pol_Latn",
    "Swedish": "swe_Latn",
    "Norwegian": "nob_Latn",
    "Danish": "dan_Latn",
    "Finnish": "fin_Latn",
    "Greek": "ell_Grek",
    "Hebrew": "heb_Hebr",
    "Thai": "tha_Thai",
    "Indonesian": "ind_Latn",
    "Malay": "zsm_Latn",
    "Swahili": "swh_Latn",
    "Marathi": "mar_Deva",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu",
    "Kannada": "kan_Knda",
    "Gujarati": "guj_Gujr",
    "Punjabi": "pan_Guru",
}


class Translator:
    """
    Translates text between languages using Facebook's NLLB-200 model.

    Supports 200+ languages. The model is loaded lazily on first use
    and cached for subsequent calls.
    """

    _instance = None  # Singleton instance

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
            cls._instance._tokenizer = None
        return cls._instance

    def _load_model(self):
        """Load the translation pipeline if not already loaded."""
        if self._model is None:
            print("[Translator] Loading model: facebook/nllb-200-distilled-600M")
            self._model = pipeline(
                "translation",
                model="facebook/nllb-200-distilled-600M",
                device_map="auto",  # Uses GPU if available, else CPU
            )
            print("[Translator] Model loaded successfully.")

    def get_language_code(self, language_name: str) -> str:
        """
        Convert a user-friendly language name to NLLB language code.

        Args:
            language_name: Human-readable language name (e.g., 'French')

        Returns:
            NLLB BCP-47 language code (e.g., 'fra_Latn')
        """
        return LANGUAGE_CODE_MAP.get(language_name, language_name)

    def translate(self, text: str, source_language: str, target_language: str) -> dict:
        """
        Translate text from source language to target language.

        Args:
            text: Input text to translate.
            source_language: Source language name or NLLB code.
            target_language: Target language name or NLLB code.

        Returns:
            dict with keys:
                - translated_text (str): The translated text.
                - source_language (str): Resolved source language code.
                - target_language (str): Resolved target language code.
        """
        self._load_model()

        if not text or not text.strip():
            return {
                "translated_text": "",
                "source_language": source_language,
                "target_language": target_language,
            }

        # Resolve friendly names to NLLB codes
        src_code = self.get_language_code(source_language)
        tgt_code = self.get_language_code(target_language)

        with __import__('torch').no_grad():   # No gradient tracking → faster
            result = self._model(
                text,
                src_lang=src_code,
                tgt_lang=tgt_code,
                max_length=200,            # Reduced from 512 → faster for typical inputs
            )

        translated = result[0]["translation_text"]

        return {
            "translated_text": translated,
            "source_language": src_code,
            "target_language": tgt_code,
        }


# Module-level singleton instance
translator = Translator()
