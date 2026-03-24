import os
import chromadb
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

CHROMA_PERSIST_DIR     = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "vqa_images")


class ChromaStore:
    """
    Persistent vector store for the dense leg.
    Stores 2048-dim image vectors from Nomic Embed Multimodal.
    Uses cosine similarity — correct metric for Nomic embeddings.
    """

    def __init__(self):
        Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB ready — '{CHROMA_COLLECTION_NAME}' "
            f"| {self.collection.count()} vectors stored"
        )

    # ── Write ────────────────────────────────────────────────

    def upsert(self, image_id: str, vector: list[float], image_path: str) -> None:
        """Insert or update a single image vector."""
        self.collection.upsert(
            ids=[image_id],
            embeddings=[vector],
            metadatas=[{
                "image_path": str(image_path),
                "filename":   Path(image_path).name,
            }],
        )
        logger.debug(f"Upserted: {image_id}")

    def upsert_batch(
        self,
        image_ids:   list[str],
        vectors:     list[list[float]],
        image_paths: list[str],
    ) -> None:
        """Batch upsert — much faster than looping upsert() individually."""
        self.collection.upsert(
            ids=image_ids,
            embeddings=vectors,
            metadatas=[
                {"image_path": str(p), "filename": Path(p).name}
                for p in image_paths
            ],
        )
        logger.info(f"Batch upserted {len(image_ids)} vectors into ChromaDB")

    # ── Read ─────────────────────────────────────────────────

    def query(self, query_vector: list[float], top_n: int = 20) -> list[dict]:
        """
        Find top_n images most similar to query_vector.
        query_vector comes from MultimodalEmbedder.embed_text(question).

        Returns:
            [{"image_id", "image_path", "filename", "score", "rank"}, ...]
        """
        count = self.collection.count()
        if count == 0:
            logger.warning("ChromaDB is empty — run indexing first")
            return []

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_n, count),
            include=["metadatas", "distances"],
        )

        output = []
        for rank, (image_id, meta, dist) in enumerate(zip(
            results["ids"][0],
            results["metadatas"][0],
            results["distances"][0],
        ), start=1):
            output.append({
                "image_id":   image_id,
                "image_path": meta["image_path"],
                "filename":   meta["filename"],
                "score":      round(1 - dist, 4),
                "rank":       rank,
            })

        logger.debug(f"Dense query returned {len(output)} results")
        return output

    # ── Utility ──────────────────────────────────────────────

    def count(self) -> int:
        return self.collection.count()

    def delete_collection(self) -> None:
        self.client.delete_collection(CHROMA_COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning(f"Collection '{CHROMA_COLLECTION_NAME}' wiped and recreated")