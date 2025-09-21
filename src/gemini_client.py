from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional

@dataclass(slots=True)
class GeminiCleaner:
    enabled: bool
    model_name: str

    def __post_init__(self) -> None:
        self._client = None
        if self.enabled:
            key = os.getenv("GEMINI_API_KEY", "").strip()
            if key:
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=key)
                    self._client = genai.GenerativeModel(self.model_name)
                except Exception:
                    self._client = None
            else:
                self.enabled = False

    def clean_plate(self, raw_text: str) -> str:
        """
        Normalize OCR output to uppercase alphanumeric, remove spaces/punct.
        If Gemini available, ask it to return best-guess normalized plate string.
        """
        # fast local normalization first
        base = "".join(ch for ch in raw_text.upper() if ch.isalnum())
        if not self.enabled or self._client is None:
            return base

        prompt = (
            "Given noisy OCR output of a vehicle license plate, return only the most "
            "likely cleaned plate text, uppercase alphanumeric, no spaces, no punctuation. "
            f"OCR: {raw_text!r}"
        )
        try:
            resp = self._client.generate_content(prompt)
            text = (resp.text or "").strip().upper()
            text = "".join(ch for ch in text if ch.isalnum())
            return text or base
        except Exception:
            return base