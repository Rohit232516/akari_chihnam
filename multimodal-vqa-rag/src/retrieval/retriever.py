from loguru import logger


class Retriever:
    """
    Hybrid Retriever:
    - Dense retrieval (Chroma)
    - Sparse retrieval (BM25)
    - Fusion using Reciprocal Rank Fusion (RRF)
    """

    def __init__(self, embedder):
        from src.retrieval.dense  import DenseRetriever
        from src.retrieval.sparse import SparseRetriever
        from src.retrieval.rrf import reciprocal_rank_fusion as rrf_fusion
        from src.indexing.chroma_store import ChromaStore
        from src.indexing.bm25_index   import BM25Index

        self.embedder = embedder
        self.chroma   = ChromaStore()
        self.bm25     = BM25Index()
        self.dense    = DenseRetriever(embedder, self.chroma)
        self.sparse   = SparseRetriever(self.bm25)
        self.rrf      = rrf_fusion

        logger.info("Retriever initialised (dense + sparse + RRF)")

    def retrieve(self, query_text: str, top_k: int = 5) -> list[dict]:
        """
        Hybrid retrieval:
        1. Dense vector search via ChromaDB
        2. Sparse keyword search via BM25
        3. Fuse with RRF
        4. Return top_k results
        """
        logger.info(f"Starting retrieval for: '{query_text[:60]}'")

        # 1. Dense retrieval
        dense_results = self.dense.retrieve(query_text, top_n=20)
        logger.info(f"Dense results: {len(dense_results)}")

        # 2. Sparse retrieval
        sparse_results = self.sparse.retrieve(query_text, top_n=20)
        logger.info(f"Sparse results: {len(sparse_results)}")

        # 3. RRF fusion
        if sparse_results:
            fused = self.rrf(dense_results, sparse_results)
            logger.info("Applied RRF fusion")
        else:
            fused = dense_results
            logger.info("Using dense only (no sparse results)")

        # 4. Top-k cutoff
        final = fused[:top_k]
        logger.info(f"Final results: {len(final)}")
        return final