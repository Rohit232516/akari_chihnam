import os
import torch
from PIL import Image
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

MODEL_ID   = os.getenv("EMBED_MODEL_ID", "nomic-ai/nomic-embed-multimodal-3b")
DEVICE     = os.getenv("EMBED_DEVICE", "cpu")
BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", 8))


class MultimodalEmbedder:
    """
    Wraps Nomic Embed Multimodal 3B.
    Loads via AutoProcessor + Qwen2_5_VLModel directly,
    bypassing the broken PEFT adapter config entirely.
    """

    def __init__(self):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        logger.info(f"Loading embedding model: {MODEL_ID} on {DEVICE}")

        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
        )

        # load the base model directly — skip PEFT adapter injection
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            device_map=DEVICE,
            ignore_mismatched_sizes=True,
        )
        self.model.eval()
        logger.info(f"Embedding model loaded — hidden size: {self.model.config.hidden_size}")

    # ── Image embedding — dense leg, index time ──────────────

    def embed_image(self, image_path: str) -> list[float]:
        """Embed a single raw image into a vector via mean pooling."""
        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            images=image,
            text="",
            return_tensors="pt",
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        vector = self._mean_pool(outputs)
        logger.debug(f"Embedded image: {Path(image_path).name} → dim {len(vector)}")
        return vector

    def embed_images_batch(self, image_paths: list[str]) -> list[list[float]]:
        """Embed images one at a time and collect results."""
        all_vectors = []
        total = len(image_paths)
        for i, path in enumerate(image_paths):
            vec = self.embed_image(path)
            all_vectors.append(vec)
            logger.info(f"Embedded {i+1}/{total} images")
        return all_vectors

    # ── Text embedding — dense leg, query time ───────────────

    def embed_text(self, text: str) -> list[float]:
        """Embed a text query via mean pooling over last hidden state."""
        prefixed = f"search_query: {text}"

        inputs = self.processor(
            text=prefixed,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        vector = self._mean_pool(outputs, inputs.get("attention_mask"))
        logger.debug(f"Embedded query: '{text[:50]}' → dim {len(vector)}")
        return vector

    # ── Helpers ──────────────────────────────────────────────

    def _mean_pool(self, outputs, attention_mask=None) -> list[float]:
        """Mean pool over last hidden state to get fixed-size vector."""
        hidden = outputs.hidden_states[-1]          # (batch, seq_len, hidden)

        if attention_mask is not None:
            mask   = attention_mask.unsqueeze(-1).float()
            summed = (hidden * mask).sum(dim=1)
            count  = mask.sum(dim=1).clamp(min=1e-9)
            vector = (summed / count)[0]
        else:
            vector = hidden[0].mean(dim=0)

        return vector.cpu().float().tolist()

    @staticmethod
    def is_supported_image(path: str) -> bool:
        return Path(path).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}