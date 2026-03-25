import os
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

from google import genai
from google.genai import types

load_dotenv()

MODEL_ID = os.getenv("EMBED_MODEL_ID", "gemini-embedding-exp-03-07")


class MultimodalEmbedder:
    """
    Multimodal embedder using Google Gemini Embedding API.
    Maps images and text into the same vector space via one API call.
    No local model, no GPU, no RAM issues.
    """

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not set in .env. "
                "Get one free at https://aistudio.google.com/app/apikey"
            )

        self.client = genai.Client(api_key=api_key)
        logger.info(f"Gemini embedder ready — model: {MODEL_ID}")

    # ── Image embedding — dense leg, index time ──────────────

    def embed_image(self, image_path: str) -> list[float]:
        """Embed a single image into a vector. Raw bytes sent directly."""
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        response = self.client.models.embed_content(
            model=MODEL_ID,
            contents=types.Part.from_bytes(
                data=image_bytes,
                mime_type=self._get_mime_type(image_path),
            ),
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
            ),
        )

        vector = list(response.embeddings[0].values)
        logger.debug(f"Embedded image: {Path(image_path).name} → dim {len(vector)}")
        return vector

    def embed_images_batch(self, image_paths: list[str]) -> list[list[float]]:
        """Embed a list of images and return vectors in same order."""
        all_vectors = []
        total = len(image_paths)
        for i, path in enumerate(image_paths):
            vector = self.embed_image(path)
            all_vectors.append(vector)
            logger.info(f"Embedded {i+1}/{total} — {Path(path).name}")
        return all_vectors

    # ── Text embedding — dense leg, query time ───────────────

    def embed_text(self, text: str) -> list[float]:
        """Embed a text query into the same vector space as images."""
        response = self.client.models.embed_content(
            model=MODEL_ID,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
            ),
        )

        vector = list(response.embeddings[0].values)
        logger.debug(f"Embedded query: '{text[:60]}' → dim {len(vector)}")
        return vector

    # ── Helpers ──────────────────────────────────────────────

    @staticmethod
    def _get_mime_type(path: str) -> str:
        return {
            ".jpg":  "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png":  "image/png",
            ".webp": "image/webp",
            ".bmp":  "image/bmp",
        }.get(Path(path).suffix.lower(), "image/jpeg")

    @staticmethod
    def is_supported_image(path: str) -> bool:
        return Path(path).suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".webp", ".bmp"
        }