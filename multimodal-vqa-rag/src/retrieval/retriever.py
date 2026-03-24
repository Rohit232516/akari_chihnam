from loguru import logger


class Retriever:
    """
    Hybrid Retriever:
    - Dense retrieval (Chroma)
    - Sparse retrieval (BM25)
    - Fusion using Reciprocal Rank Fusion (RRF)
    """

    def __init__(self, embedder):
        # Imports (kept inside to avoid circular deps)
        from src.retrieval.dense import DenseRetriever
        from src.retrieval.sparse import SparseRetriever
        from src.retrieval.rrf import reciprocal_rank_fusion
        from src.indexing.chroma_store import ChromaStore
        from src.indexing.bm25_index import BM25Index

        self.embedder = embedder

        # Datastores
        self.chroma = ChromaStore()
        self.bm25 = BM25Index()

        # Retrievers
        self.dense = DenseRetriever(embedder, self.chroma)
        self.sparse = SparseRetriever(self.bm25)

        # Fusion
        self.rrf = reciprocal_rank_fusion

        logger.info("Retriever initialised (dense + sparse + RRF)")

    # ─────────────────────────────────────────────
    # RETRIEVE
    # ─────────────────────────────────────────────
    def retrieve(self, query_vector, query_text=None, top_k=5):
        """
        Hybrid retrieval:
        1. Dense (vector search)
        2. Sparse (BM25)
        3. Fuse using RRF
        """

        logger.info("Starting retrieval...")

        # 1. Dense retrieval
        dense_results = self.dense.retrieve(query_vector, top_k=top_k)
        logger.info(f"Dense results: {len(dense_results)}")

        # 2. Sparse retrieval
        sparse_results = []
        if query_text:
            sparse_results = self.sparse.retrieve(query_text, top_k=top_k)
            logger.info(f"Sparse results: {len(sparse_results)}")

        # 3. Fusion (RRF)
        if sparse_results:
            results = self.rrf(dense_results, sparse_results)
            logger.info("Applied RRF fusion")
        else:
            results = dense_results
            logger.info("Using dense only")

        # 4. Top-K cutoff
        final_results = results[:top_k]
        logger.info(f"Final results: {len(final_results)}")

        return final_results