import os
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

RRF_K = int(os.getenv("RRF_K", 60))


def reciprocal_rank_fusion(
    dense_results:  list[dict],
    sparse_results: list[dict],
    k:              int | None = None,
) -> list[dict]:
    """
    Reciprocal Rank Fusion — merges dense and sparse ranked lists.

    WHY RRF:
    Dense scores (cosine similarity 0-1) and sparse scores (BM25 tf-idf)
    live on completely different scales. You cannot simply add them.
    RRF sidesteps this by using only the RANK POSITION of each result,
    not its score. This makes fusion robust and parameter-free.

    FORMULA:
    For each image that appears in either list:
        rrf_score = sum(1 / (k + rank_in_list))
    where k=60 is a standard constant that dampens the impact of top ranks.

    An image appearing at rank 1 in both lists scores:
        1/(60+1) + 1/(60+1) = 0.0328
    An image appearing at rank 1 in only one list scores:
        1/(60+1) = 0.0164

    So images that both legs agree on float to the top naturally.

    Args:
        dense_results:  output of DenseRetriever.retrieve()
        sparse_results: output of SparseRetriever.retrieve()
        k:              RRF constant (default 60, rarely needs changing)

    Returns:
        Merged list sorted by fused RRF score descending.
        Each item: {"image_id", "rrf_score", "rank",
                    "in_dense", "in_sparse",
                    "dense_rank", "sparse_rank"}
    """
    k = k or RRF_K

    scores      = {}   # image_id → cumulative rrf score
    dense_rank  = {}   # image_id → rank in dense list
    sparse_rank = {}   # image_id → rank in sparse list

    # accumulate scores from dense list
    for item in dense_results:
        img_id = item["image_id"]
        rank   = item["rank"]
        scores[img_id]     = scores.get(img_id, 0.0) + 1.0 / (k + rank)
        dense_rank[img_id] = rank

    # accumulate scores from sparse list
    for item in sparse_results:
        img_id = item["image_id"]
        rank   = item["rank"]
        scores[img_id]      = scores.get(img_id, 0.0) + 1.0 / (k + rank)
        sparse_rank[img_id] = rank

    # sort by fused score descending
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for out_rank, (img_id, rrf_score) in enumerate(fused, start=1):
        results.append({
            "image_id":    img_id,
            "rrf_score":   round(rrf_score, 6),
            "rank":        out_rank,
            "in_dense":    img_id in dense_rank,
            "in_sparse":   img_id in sparse_rank,
            "dense_rank":  dense_rank.get(img_id),
            "sparse_rank": sparse_rank.get(img_id),
        })

    logger.info(
        f"RRF fusion: {len(dense_results)} dense + {len(sparse_results)} sparse "
        f"→ {len(results)} unique candidates"
    )
    return results