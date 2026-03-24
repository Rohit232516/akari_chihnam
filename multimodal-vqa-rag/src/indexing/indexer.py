import os
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

from src.indexing.embed        import MultimodalEmbedder
from src.indexing.chroma_store import ChromaStore
from src.indexing.bm25_index   import BM25Index

load_dotenv()

IMAGE_DIR = os.getenv("IMAGE_DIR", "./data/images")


class Indexer:
    """
    Orchestrates the full indexing pipeline. Run once per image corpus.

    DENSE LEG — raw image pixels → Nomic Embed Multimodal → vector → ChromaDB
        No text. No captions. The image IS the embedding input.

    SPARSE LEG — image → moondream generates caption → BM25 keyword index
        Captions are ONLY for BM25. Never used for embedding.

    At query time (handled by retrieval/ modules):
        question → Nomic embed_text() → cosine search ChromaDB   (dense)
        question → BM25 keyword search over captions              (sparse)
                 → RRF fusion → BGE reranker → top-4 images
                 → moondream answers with raw images + question
    """

    def __init__(self):
        self.embedder = MultimodalEmbedder()
        self.chroma   = ChromaStore()
        self.bm25     = BM25Index()

    def index_directory(
        self,
        image_dir:     str | None = None,
        force_reindex: bool = False,
    ) -> None:
        """
        Index all supported images in image_dir.

        Args:
            image_dir:     path to image folder (defaults to IMAGE_DIR in .env)
            force_reindex: wipe ChromaDB and regenerate everything if True
        """
        image_dir = Path(image_dir or IMAGE_DIR)

        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        image_paths = [
            str(p) for p in sorted(image_dir.iterdir())
            if MultimodalEmbedder.is_supported_image(str(p))
        ]

        if not image_paths:
            logger.warning(f"No supported images found in {image_dir}")
            return

        image_ids = [Path(p).stem for p in image_paths]
        logger.info(f"Found {len(image_paths)} images in {image_dir}")

        if force_reindex:
            logger.warning("force_reindex=True — wiping ChromaDB")
            self.chroma.delete_collection()

        # ── DENSE LEG ────────────────────────────────────────
        # Raw images → Nomic Embed Multimodal → vectors → ChromaDB
        # Pure multimodal embedding. No text anywhere in this leg.
        logger.info("DENSE LEG — embedding raw images with Nomic Embed Multimodal 3B")
        vectors = self.embedder.embed_images_batch(image_paths)
        self.chroma.upsert_batch(
            image_ids=image_ids,
            vectors=vectors,
            image_paths=image_paths,
        )
        logger.info(f"Dense leg done — {self.chroma.count()} vectors in ChromaDB")

        # ── SPARSE LEG ───────────────────────────────────────
        # Images → moondream captions → BM25 keyword index
        # Captions are ONLY for BM25. Never used for embeddings.
        logger.info("SPARSE LEG — generating captions with moondream for BM25")
        captions_dict = self._generate_captions(image_paths)
        self.bm25.build(captions_dict)
        logger.info(f"Sparse leg done — {self.bm25.count()} captions in BM25")

        logger.info("Indexing complete — hybrid search ready")
        logger.info(f"  Dense : {self.chroma.count()} image vectors in ChromaDB")
        logger.info(f"  Sparse: {self.bm25.count()} captions in BM25")

    # ── Caption generation — sparse leg only ─────────────────

    def _generate_captions(self, image_paths: list[str]) -> dict[str, str]:
        """
        Generate one short caption per image using moondream.
        Captions are saved to data/captions/captions.json — runs once only.

        Returns: {"img_001": "three red cars near a building", ...}
        """
        import ollama

        OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL", "moondream")
        CAPTION_PROMPT = (
            "Describe this image briefly. "
            "List the objects, their count, colours, and positions."
        )

        captions = {}
        total    = len(image_paths)

        for i, path in enumerate(image_paths):
            image_id = Path(path).stem
            try:
                response = ollama.chat(
                    model=OLLAMA_MODEL,
                    messages=[{
                        "role":    "user",
                        "content": CAPTION_PROMPT,
                        "images":  [path],
                    }],
                    options={"temperature": 0.1},
                )
                caption = response["message"]["content"].strip()
            except Exception as e:
                logger.warning(f"Caption failed for {image_id}: {e} — using filename")
                caption = image_id.replace("_", " ")

            captions[image_id] = caption
            logger.info(f"[{i+1}/{total}] {image_id}: {caption[:80]}")

        return captions