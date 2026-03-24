import os
import torch
from PIL import Image
from pathlib import Path
from loguru import logger
from transformers import AutoProcessor, AutoModel
from dotenv import load_dotenv

load_dotenv()

MODEL_ID   = os.getenv("EMBED_MODEL_ID", "nomic-ai/nomic-embed-multimodal-3b")
DEVICE     = os.getenv("EMBED_DEVICE", "cpu")
BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", 8))


class MultimodalEmbedder:
    """
    Wraps Nomic Embed Multimodal 3B.

    embed_image()        — raw image pixels → 2048-dim vector  (index time)
    embed_images_batch() — batch of images  → list of vectors  (index time)
    embed_text()         — query text       → 2048-dim vector  (query time)

    Images and text share the same vector space, so cosine similarity
    directly measures how well an image matches a text question.
    No captions, no object detection, no preprocessing needed.
    """

    def __init__(self):
        logger.info(f"Loading embedding model: {MODEL_ID} on {DEVICE}")
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model     = AutoModel.from_pretrained(MODEL_ID).to(DEVICE)
        self.model.eval()
        logger.info("Embedding model loaded")

    # ── Image embedding — dense leg, index time ──────────────

    def embed_image(self, image_path: str) -> list[float]:
        """Embed a single raw image into a 2048-dim vector."""
        image  = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
        vector = outputs[0].cpu().float().tolist()
        logger.debug(f"Embedded image: {Path(image_path).name} → dim {len(vector)}")
        return vector

    def embed_images_batch(self, image_paths: list[str]) -> list[list[float]]:
        """Embed a list of images in batches. Returns vectors in same order as input."""
        all_vectors = []
        total       = len(image_paths)

        for i in range(0, total, BATCH_SIZE):
            batch_paths  = image_paths[i : i + BATCH_SIZE]
            batch_images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = self.processor(
                images=batch_images,
                return_tensors="pt",
                padding=True,
            ).to(DEVICE)
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
            all_vectors.extend(outputs.cpu().float().tolist())
            logger.info(f"Embedded {min(i + BATCH_SIZE, total)}/{total} images")

        return all_vectors

    # ── Text embedding — dense leg, query time ───────────────

    def embed_text(self, text: str) -> list[float]:
        """
        Embed a text query into the same 2048-dim space as images.
        Only call this at query time — never for indexing.
        Nomic requires 'search_query:' prefix for retrieval mode.
        """
        prefixed = f"search_query: {text}"
        inputs   = self.processor(text=prefixed, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
        vector = outputs[0].cpu().float().tolist()
        logger.debug(f"Embedded query: '{text[:50]}' → dim {len(vector)}")
        return vector

    @staticmethod
    def is_supported_image(path: str) -> bool:
        return Path(path).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}