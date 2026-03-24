import os
from loguru import logger
from dotenv import load_dotenv

from src.indexing.embed        import MultimodalEmbedder
from src.indexing.chroma_store import ChromaStore

load_dotenv()

DENSE_TOP_N = int(os.getenv("DENSE_TOP_N", 20))


class DenseRetriever:
    """
    Dense retrieval leg.

    Embeds the user's text question using Nomic Embed Multimodal
    and searches ChromaDB for the most visually similar images.

    Because images and text share the same vector space, this
    directly measures how well each image's visual content
    matches the meaning of the question — no captions needed.
    """

    def __init__(self, embedder: MultimodalEmbedder, chroma: ChromaStore):
        self.embedder = embedder
        self.chroma   = chroma

    def retrieve(self, question: str, top_n: int | None = None) -> list[dict]:
        """
        Retrieve top_n images most semantically similar to the question.

        Args:
            question: the user's VQA question as plain text
            top_n:    number of results (defaults to DENSE_TOP_N in .env)

        Returns:
            [
                {
                    "image_id":   "img_001",
                    "image_path": "./data/images/img_001.jpg",
                    "filename":   "img_001.jpg",
                    "score":      0.91,
                    "rank":       1,
                    "source":     "dense",
                },
                ...
            ]
        """
        top_n = top_n or DENSE_TOP_N

        logger.debug(f"Dense retrieval for: '{question[:60]}'")

        query_vector = self.embedder.embed_text(question)
        results      = self.chroma.query(query_vector, top_n=top_n)

        for r in results:
            r["source"] = "dense"

        logger.info(f"Dense leg returned {len(results)} results")
        return results