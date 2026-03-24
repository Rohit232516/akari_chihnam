import os
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

from google import genai
from google.genai import types

load_dotenv()

MODEL_ID = os.getenv("EMBED_MODEL_ID", "gemini-embedding-2-preview")


class MultimodalEmbedder:
    """
    Gemini NEW SDK embedder (supports gemini-embedding-2-preview)
    """

    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")

        self.client = genai.Client(api_key=api_key)
        logger.info(f"Using Gemini model: {MODEL_ID}")

    # ── Image embedding ──────────────────────────────────────

    def embed_image(self, image_path: str) -> list[float]:
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        

        response = self.client.models.embed_content(
            model=MODEL_ID,
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=self._get_mime_type(image_path),
                )
            ]
        )
        return response.embeddings[0].values

    def embed_images_batch(self, image_paths: list[str]) -> list[list[float]]:
        return [self.embed_image(p) for p in image_paths]

    # ── Text embedding ───────────────────────────────────────

    def embed_text(self, text: str) -> list[float]:
        response = self.client.models.embed_content(
            model=MODEL_ID,
            contents=[text]
        )
        return response.embeddings[0].values

    # ── Helpers ──────────────────────────────────────────────

    @staticmethod
    def _get_mime_type(path: str) -> str:
        ext = Path(path).suffix.lower()
        return {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }.get(ext, "image/jpeg")

    @staticmethod
    def is_supported_image(path: str) -> bool:
        return Path(path).suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".webp", ".bmp"
        }