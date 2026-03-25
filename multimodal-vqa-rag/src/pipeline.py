import argparse
import time
from loguru import logger

from src.indexing.embed    import MultimodalEmbedder
from src.indexing.indexer  import Indexer
from src.retrieval.retriever import Retriever
from src.retrieval.rerank  import Reranker
from src.generation.vqa    import VQAGenerator

TOP_K = 4  # number of candidates passed to the LLM


class VQAPipeline:
    def __init__(self):
        logger.info("Initialising VQA pipeline...")

        self.embedder  = MultimodalEmbedder()
        self.indexer   = Indexer(self.embedder)
        self.retriever = Retriever(self.embedder)
        self.reranker  = Reranker()
        self.generator = VQAGenerator()

        logger.info("Pipeline ready")

    # ── Index ─────────────────────────────────────

    def index(self, image_dir: str, force_reindex: bool = False):
        start = time.time()
        logger.info(f"Indexing images from: {image_dir}")
        self.indexer.index_directory(
            image_dir=image_dir,
            force_reindex=force_reindex,
        )
        logger.info(f"Indexing complete in {time.time() - start:.2f}s")

    # ── Query ─────────────────────────────────────

    def query(self, question: str, image_path: str = None) -> str:
        """
        Full pipeline:
        1. Retrieve top candidates via hybrid search (dense + sparse + RRF)
        2. Rerank with cross-encoder
        3. Generate answer with Gemini 2.5 Flash
        """
        total_start = time.time()

        # 1. Retrieve
        t1 = time.time()
        candidates = self.retriever.retrieve(
            query_text=question,
            top_k=TOP_K,
        )
        logger.info(f"[Retrieve] {len(candidates)} candidates in {time.time() - t1:.2f}s")

        if not candidates:
            return "No relevant images found. Please index some images first."

        # 2. Rerank
        t2 = time.time()
        reranked = self.reranker.rerank(
            question=question,
            candidates=candidates,
            bm25_index=self.retriever.bm25,
            top_k=TOP_K,
        )
        logger.info(f"[Rerank] Done in {time.time() - t2:.2f}s")

        # extract image paths from reranked results
        image_paths = []
        for item in reranked:
            img_id  = item.get("image_id", "")
            # look up path from chroma metadata
            results = self.retriever.chroma.query(
                self.embedder.embed_text(question), top_n=TOP_K
            )
            for r in results:
                if r["image_id"] == img_id:
                    image_paths.append(r["image_path"])
                    break

        # if uploaded image was provided, prepend it
        if image_path:
            image_paths.insert(0, image_path)

        # 3. Generate
        t3 = time.time()
        answer = self.generator.generate(
            question=question,
            image_paths=image_paths if image_paths else None,
        )
        logger.info(f"[Generate] Done in {time.time() - t3:.2f}s")
        logger.info(f"[Total] {time.time() - total_start:.2f}s")

        return answer


# ── CLI ───────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multimodal VQA RAG Pipeline")
    parser.add_argument("--index",     action="store_true", help="Index images")
    parser.add_argument("--image-dir", type=str,            help="Directory of images to index")
    parser.add_argument("--query",     type=str,            help="Ask a question")
    parser.add_argument("--image",     type=str,            help="Optional image path for query")
    parser.add_argument("--reindex",   action="store_true", help="Force re-index")
    args = parser.parse_args()

    pipeline = VQAPipeline()

    if args.index:
        if not args.image_dir:
            raise ValueError("--image-dir required for indexing")
        pipeline.index(args.image_dir, force_reindex=args.reindex)

    elif args.query:
        answer = pipeline.query(args.query, image_path=args.image)
        print("\nAnswer:\n", answer)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()