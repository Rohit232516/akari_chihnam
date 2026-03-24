import os
from loguru import logger
from dotenv import load_dotenv

from src.indexing.bm25_index import BM25Index

load_dotenv()

SPARSE_TOP_N = int(os.getenv("SPARSE_TOP_N", 20))


class SparseRetriever:
    """
    Sparse retrieval leg.

    Keyword searches the BM25 index built over image captions.
    Catches exact word matches that dense vectors average away —
    specific object names, counts, colours, spatial positions.
    """

    def __init__(self, bm25: BM25Index):
        self.bm25 = bm25

    def retrieve(self, question: str, top_n: int | None = None) -> list[dict]:
        """
        Retrieve top_n images by keyword match against captions.

        Args:
            question: the user's VQA question as plain text
            top_n:    number of results (defaults to SPARSE_TOP_N in .env)

        Returns:
            [
                {
                    "image_id": "img_001",
                    "score":    4.21,
                    "rank":     1,
                    "source":   "sparse",
                },
                ...
            ]
        """
        top_n = top_n or SPARSE_TOP_N

        if not self.bm25.is_ready():
            logger.warning("BM25 index not ready — sparse leg returning empty")
            return []

        logger.debug(f"Sparse retrieval for: '{question[:60]}'")

        results = self.bm25.query(question, top_n=top_n)

        for r in results:
            r["source"] = "sparse"

        logger.info(f"Sparse leg returned {len(results)} results")
        return results