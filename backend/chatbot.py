"""
Chatbot Module (Optimised)
==========================
Uses facebook/blenderbot-400M-distill with greedy decoding
for significantly faster response generation.

Optimisation applied:
  num_beams=1  (greedy decoding)
    - Was: num_beams=4 → considers 4 candidate sequences simultaneously
    - Now: num_beams=1 → picks the single best token at each step
    - Speed improvement: ~3–4× faster generation
    - Quality trade-off: minimal for conversational dialogue
"""

from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import torch

MODEL_NAME = "facebook/blenderbot-400M-distill"


class Chatbot:
    """
    Generates conversational AI responses using Facebook's BlenderBot model.
    Singleton pattern — model loaded once and reused across all requests.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
            cls._instance._tokenizer = None
        return cls._instance

    def _load_model(self):
        """Load BlenderBot tokeniser and model in eval mode."""
        if self._model is None:
            print(f"[Chatbot] Loading model: {MODEL_NAME}")
            self._tokenizer = BlenderbotTokenizer.from_pretrained(MODEL_NAME)
            self._model = BlenderbotForConditionalGeneration.from_pretrained(MODEL_NAME)
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(self._device)
            self._model.eval()  # Disable dropout → faster + deterministic
            print(f"[Chatbot] Model loaded on {self._device}.")

    def respond(self, text: str, language: str = "en") -> dict:
        """
        Generate a conversational AI response.

        Args:
            text: User input message.
            language: Language code (informational).

        Returns:
            dict with keys:
                - response (str): AI-generated reply.
                - model (str): Model name.
        """
        self._load_model()

        if not text or not text.strip():
            return {"response": "Please provide some text for me to respond to.", "model": MODEL_NAME}

        inputs = self._tokenizer(
            [text],
            return_tensors="pt",
            truncation=True,
            max_length=128,
        ).to(self._device)

        with torch.no_grad():          # No gradient tracking → faster + less RAM
            reply_ids = self._model.generate(
                **inputs,
                max_new_tokens=80,     # Reduced from 100/128 for faster generation
                num_beams=1,           # ← GREEDY decoding: ~3-4× faster than num_beams=4
                do_sample=False,       # Deterministic output
            )

        response = self._tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
        return {"response": response.strip(), "model": MODEL_NAME}


# Module-level singleton
chatbot = Chatbot()
