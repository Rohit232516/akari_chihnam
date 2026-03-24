import os
import time
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


class VQAPipeline:
    """
    End-to-end Multimodal VQA RAG Pipeline.

    Wires all sprints together into two public methods:

        pipeline.index(image_dir)          — Sprint 1 (indexing)
        pipeline.query(question)           — Sprint 2 + 3 (retrieval + generation)

    Full flow on query():
        question
          ├── DenseRetriever  → embed_text → cosine search ChromaDB  (Sprint 2)
          ├── SparseRetriever → BM25 keyword search over captions     (Sprint 2)
          │
          └── rrf_fusion(dense, sparse)                               (Sprint 2)
                    ↓
              Reranker.rerank() → top-4 images                        (Sprint 2)
                    ↓
              VQAGenerator.generate(question, top-4 images)           (Sprint 3)
                    ↓
              answer string
    """

    def __init__(self):
        logger.info("Initialising VQA pipeline...")

        # Sprint 1 — indexing components
        from src.indexing.embed        import MultimodalEmbedder
        from src.indexing.chroma_store import ChromaStore
        from src.indexing.bm25_index   import BM25Index
        from src.indexing.indexer      import Indexer

        # Sprint 2 — retrieval components
        from src.retrieval.dense  import DenseRetriever
        from src.retrieval.sparse import SparseRetriever
        from src.retrieval.rrf    import rrf_fusion
        from src.retrieval.rerank import Reranker

        # Sprint 3 — generation component
        from src.generation.vqa import VQAGenerator

        # shared instances
        self.embedder = MultimodalEmbedder()
        self.chroma   = ChromaStore()
        self.bm25     = BM25Index()

        # wire retrieval
        self.dense_retriever  = DenseRetriever(self.embedder, self.chroma)
        self.sparse_retriever = SparseRetriever(self.bm25)
        self.rrf_fusion       = rrf_fusion
        self.reranker         = Reranker()

        # generation
        self.vqa = VQAGenerator()

        # pass shared instances into Indexer — no second model load
        self.indexer = Indexer(
            embedder=self.embedder,
            chroma=self.chroma,
            bm25=self.bm25,
        )

        logger.info("Pipeline ready")

    # ── Public: index ────────────────────────────────────────

    def index(
        self,
        image_dir:     str | None = None,
        force_reindex: bool = False,
    ) -> None:
        """
        Index all images in image_dir.
        Run once per corpus — or with force_reindex=True to rebuild.

        Args:
            image_dir:     path to image folder (defaults to IMAGE_DIR in .env)
            force_reindex: wipe and rebuild from scratch if True
        """
        logger.info(f"Indexing: {image_dir or 'default IMAGE_DIR'}")
        self.indexer.index_directory(
            image_dir=image_dir,
            force_reindex=force_reindex,
        )

    # ── Public: query ────────────────────────────────────────

    def query(
        self,
        question:   str,
        dense_top_n:  int = 20,
        sparse_top_n: int = 20,
        final_top_k:  int = 4,
    ) -> dict:
        """
        Run the full retrieval + generation pipeline for a question.

        Args:
            question:     the VQA question
            dense_top_n:  candidates from dense search
            sparse_top_n: candidates from sparse search
            final_top_k:  images passed to the LLM after reranking

        Returns:
            {
                "question":      "How many cars are there?",
                "answer":        "There are three red cars...",
                "top_images":    ["path/img1.jpg", "path/img2.jpg", ...],
                "timing": {
                    "dense_ms":   45,
                    "sparse_ms":  3,
                    "rrf_ms":     1,
                    "rerank_ms":  380,
                    "generate_ms": 4200,
                    "total_ms":   4629,
                }
            }
        """
        logger.info(f"Query: '{question}'")
        timing = {}
        t0     = time.time()

        # ── Step 1: dense retrieval ───────────────────────────
        t = time.time()
        dense_results = self.dense_retriever.retrieve(question, top_n=dense_top_n)
        timing["dense_ms"] = int((time.time() - t) * 1000)

        # ── Step 2: sparse retrieval ──────────────────────────
        t = time.time()
        sparse_results = self.sparse_retriever.retrieve(question, top_n=sparse_top_n)
        timing["sparse_ms"] = int((time.time() - t) * 1000)

        # ── Step 3: RRF fusion ────────────────────────────────
        t = time.time()
        fused = self.rrf_fusion(dense_results, sparse_results)
        timing["rrf_ms"] = int((time.time() - t) * 1000)

        if not fused:
            logger.warning("No results from fusion — index may be empty")
            return {
                "question":   question,
                "answer":     "No images indexed yet. Run pipeline.index() first.",
                "top_images": [],
                "timing":     timing,
            }

        # ── Step 4: rerank ────────────────────────────────────
        t = time.time()
        reranked = self.reranker.rerank(
            question=question,
            candidates=fused,
            bm25_index=self.bm25,
            top_k=final_top_k,
        )
        timing["rerank_ms"] = int((time.time() - t) * 1000)

        # extract image paths for the LLM
        # we need to resolve paths from chroma metadata for dense-only results
        top_image_paths = self._resolve_image_paths(reranked, dense_results)

        # ── Step 5: VQA generation ────────────────────────────
        t = time.time()
        answer = self.vqa.generate(
            question=question,
            image_paths=top_image_paths,
        )
        timing["generate_ms"] = int((time.time() - t) * 1000)

        timing["total_ms"] = int((time.time() - t0) * 1000)

        # log timing summary
        logger.info(
            f"Pipeline timing — "
            f"dense: {timing['dense_ms']}ms | "
            f"sparse: {timing['sparse_ms']}ms | "
            f"rrf: {timing['rrf_ms']}ms | "
            f"rerank: {timing['rerank_ms']}ms | "
            f"generate: {timing['generate_ms']}ms | "
            f"total: {timing['total_ms']}ms"
        )

        return {
            "question":   question,
            "answer":     answer,
            "top_images": top_image_paths,
            "timing":     timing,
        }

    # ── Helpers ───────────────────────────────────────────────

    def _resolve_image_paths(
        self,
        reranked:      list[dict],
        dense_results: list[dict],
    ) -> list[str]:
        """
        Map reranked image_ids back to file paths.
        Paths come from ChromaDB metadata stored during indexing.
        """
        # build id → path lookup from dense results (which have full metadata)
        id_to_path = {r["image_id"]: r["image_path"] for r in dense_results}

        paths = []
        for item in reranked:
            img_id = item["image_id"]
            path   = id_to_path.get(img_id)
            if path and os.path.exists(path):
                paths.append(path)
            else:
                logger.warning(f"Could not resolve path for image_id: {img_id}")

        return paths


# ── CLI entry point ───────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multimodal VQA RAG Pipeline")
    parser.add_argument("--index",     action="store_true", help="Index images")
    parser.add_argument("--image-dir", type=str, default=None, help="Image directory")
    parser.add_argument("--query",     type=str, default=None, help="Ask a question")
    parser.add_argument("--reindex",   action="store_true", help="Force re-index")
    args = parser.parse_args()

    pipeline = VQAPipeline()

    if args.index:
        pipeline.index(image_dir=args.image_dir, force_reindex=args.reindex)

    if args.query:
        result = pipeline.query(args.query)
        print("\n" + "=" * 60)
        print(f"Q: {result['question']}")
        print(f"A: {result['answer']}")
        print(f"\nRetrieved images:")
        for p in result["top_images"]:
            print(f"  {p}")
        print(f"\nTiming: {result['timing']['total_ms']}ms total")
        print("=" * 60)