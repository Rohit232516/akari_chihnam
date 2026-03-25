import os
import base64
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
from google import genai
from google.genai import types

load_dotenv()

SYSTEM_PROMPT = """You are an expert visual analyst.
You will be shown one or more images and asked a question about them.
Answer accurately based only on what you can observe.
Focus on: object counts, spatial locations, and relationships between objects.
Be specific and concise."""


class VQAGenerator:
    """
    Generates answers using Gemini 2.5 Flash multimodal model.
    Accepts a question + list of image paths.
    """

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not set in .env. "
                "Get one at https://aistudio.google.com/app/apikey"
            )
        self.client = genai.Client(api_key=api_key)
        self.model  = os.getenv("VQA_MODEL", "gemini-2.5-flash")
        logger.info(f"VQAGenerator ready — model: {self.model}")

    def generate(
        self,
        question:    str,
        image_paths: list[str] | None = None,
        context:     str = "",
    ) -> str:
        """
        Generate a VQA answer.

        Args:
            question:    user's question
            image_paths: list of image file paths to include
            context:     optional text context from retrieval
        """
        try:
            parts = []

            # add system + context + question as text
            prompt = SYSTEM_PROMPT
            if context:
                prompt += f"\n\nContext:\n{context}"
            prompt += f"\n\nQuestion: {question}"
            parts.append(prompt)

            # add images
            if image_paths:
                for path in image_paths:
                    if not path or not Path(path).exists():
                        logger.warning(f"Image not found, skipping: {path}")
                        continue
                    with open(path, "rb") as f:
                        image_bytes = f.read()
                    mime = self._mime(path)
                    parts.append(
                        types.Part.from_bytes(data=image_bytes, mime_type=mime)
                    )

            response = self.client.models.generate_content(
                model=self.model,
                contents=parts,
            )
            return response.text

        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"❌ Error: {str(e)}"

    @staticmethod
    def _mime(path: str) -> str:
        return {
            ".jpg":  "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png":  "image/png",
            ".webp": "image/webp",
            ".bmp":  "image/bmp",
        }.get(Path(path).suffix.lower(), "image/jpeg")