import argparse
import time
from loguru import logger

from src.indexing.embed import MultimodalEmbedder
from src.indexing.indexer import Indexer
from src.retrieval.retriever import Retriever
from src.retrieval.rerank import Reranker
from src.generation.vqa import VQAGenerator


class VQAPipeline:
    def __init__(self):
        logger.info("Initialising VQA pipeline...")

        self.embedder = MultimodalEmbedder()
        self.indexer  = Indexer(self.embedder)
        self.retriever = Retriever(self.embedder)
        self.reranker  = Reranker()
        self.generator = VQAGenerator()

        logger.info("Pipeline ready")

    # ─────────────────────────────────────────────
    # INDEX
    # ─────────────────────────────────────────────
    def index(self, image_dir: str, force_reindex: bool = False):
        start = time.time()
        logger.info(f"Indexing images from: {image_dir}")

        self.indexer.index_directory(
            image_dir=image_dir,
            force_reindex=force_reindex
        )

        logger.info(f"Indexing complete in {time.time() - start:.2f}s")

    # ─────────────────────────────────────────────
    # QUERY
    # ─────────────────────────────────────────────
    def query(self, question: str, image_path: str = None):
        total_start = time.time()

        # 1. Embed query
        t0 = time.time()
        query_vec = self.embedder.embed_text(question)
        logger.info(f"[Embed] Done in {time.time() - t0:.2f}s")

        # 2. Retrieve
        t1 = time.time()
        candidates = self.retriever.retrieve(
            query_vector=query_vec,
            query_text=question
        )
        logger.info(f"[Retrieve] Found {len(candidates)} in {time.time() - t1:.2f}s")

        # 3. Rerank
        t2 = time.time()
        reranked = self.reranker.rerank(question, candidates)
        logger.info(f"[Rerank] Done in {time.time() - t2:.2f}s")

        # 4. Generate
        t3 = time.time()
        answer = self.generator.generate(
            question=question,
            contexts=reranked,
            image_path=image_path
        )
        logger.info(f"[Generate] Done in {time.time() - t3:.2f}s")

        logger.info(f"[Total] {time.time() - total_start:.2f}s")

        return answer


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Multimodal VQA RAG Pipeline")

    parser.add_argument("--index", action="store_true", help="Index images")
    parser.add_argument("--image-dir", type=str, help="Directory of images")

    parser.add_argument("--query", type=str, help="Ask a question")
    parser.add_argument("--image", type=str, help="Optional image for query")

    parser.add_argument("--reindex", action="store_true", help="Force reindex")

    args = parser.parse_args()

    pipeline = VQAPipeline()

    if args.index:
        if not args.image_dir:
            raise ValueError("--image-dir required for indexing")
        pipeline.index(args.image_dir, force_reindex=args.reindex)

    elif args.query:
        answer = pipeline.query(args.query, image_path=args.image)
        print("\n🧠 Answer:\n", answer)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()