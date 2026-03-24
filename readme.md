# Multimodal VQA RAG Pipeline

> A fully free, locally-run Visual Question Answering system using multimodal embeddings, hybrid search, RRF fusion, cross-encoder re-ranking, and Qwen2.5-VL via Ollama. Inspired by the paper *"Enhanced Multimodal RAG-LLM for Accurate Visual Question Answering"* (Xue et al., Dec 2024).

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Tech Stack](#tech-stack)
4. [Project Structure](#project-structure)
5. [Sprints](#sprints)
   - [Sprint 0 — Environment Setup](#sprint-0--environment-setup)
   - [Sprint 1 — Multimodal Embedding + Indexing](#sprint-1--multimodal-embedding--indexing)
   - [Sprint 2 — Hybrid Search + RRF Fusion](#sprint-2--hybrid-search--rrf-fusion)
   - [Sprint 3 — Cross-Encoder Re-ranking](#sprint-3--cross-encoder-re-ranking)
   - [Sprint 4 — VQA Generation via Ollama](#sprint-4--vqa-generation-via-ollama)
   - [Sprint 5 — Pipeline Orchestration + CLI](#sprint-5--pipeline-orchestration--cli)
   - [Sprint 6 — Evaluation + Benchmarking](#sprint-6--evaluation--benchmarking)
   - [Sprint 7 — Streamlit UI (Optional)](#sprint-7--streamlit-ui-optional)
6. [Environment Variables](#environment-variables)
7. [How to Run](#how-to-run)
8. [Datasets](#datasets)
9. [Cost](#cost)

---

## Project Overview

This project rebuilds the VQA RAG pipeline from scratch using only **free, open-source tools** — no paid APIs, no cloud subscriptions. The key improvement over the original paper is eliminating the entire Faster-RCNN + PENET preprocessing stack and replacing it with a single multimodal embedding model that directly encodes raw images.

### What it does

- Takes a dataset of images (e.g. VG-150, AUG aerial scenes, or your own)
- Embeds each image directly using **Nomic Embed Multimodal 3B**
- Stores dense vectors in **ChromaDB** and builds a **BM25 sparse index** on image captions
- At query time: runs hybrid search, fuses results with **RRF**, re-ranks with **BGE-reranker-base**
- Passes the top-4 retrieved images + user question to **Qwen2.5-VL 7B** via Ollama for a grounded VQA answer

### Improvements over the original paper

| Dimension | Original paper | This project |
|---|---|---|
| Image handling | Faster-RCNN + PENET → text chunks | Direct image embedding (Nomic) |
| Embedding model | Text2Vec Multilingual (text-only) | Nomic Embed Multimodal 3B (image + text) |
| Retrieval | Cosine similarity · top-4 only | Hybrid search (dense + BM25) + RRF + rerank |
| Vector store | In-memory (no persistence) | ChromaDB (persistent on disk) |
| LLM | Qwen-2-72B-Instruct (cloud) | Qwen2.5-VL 7B via Ollama (local, free) |
| Hardware | RTX 3060 required for all models | RTX 3060 / CPU / Google Colab — all free |
| Cost | Local GPU + potential API cost | $0 |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  PHASE A · INDEXING (run once per image corpus)          │
│                                                          │
│  Images ──► Nomic Embed Multimodal 3B                   │
│                     │                                    │
│          ┌──────────┴──────────┐                        │
│          ▼                     ▼                        │
│   Dense vectors           BM25 sparse index             │
│   (2048-dim)              (rank-bm25)                   │
│          └──────────┬──────────┘                        │
│                     ▼                                    │
│               ChromaDB (local persistent)               │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  PHASE B · QUERY (per user question)                     │
│                                                          │
│  User question ──► Nomic Embed (RETRIEVAL_QUERY mode)   │
│                         │                               │
│             ┌───────────┴───────────┐                   │
│             ▼                       ▼                   │
│     Dense top-N               Sparse top-N              │
│     (ChromaDB)                (BM25)                    │
│             └───────────┬───────────┘                   │
│                         ▼                               │
│                  RRF fusion (k=60)                      │
│                         ▼                               │
│             BGE-reranker-base (cross-encoder)           │
│                         ▼                               │
│                   Top-4 images                          │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  PHASE C · GENERATION                                    │
│                                                          │
│  Top-4 images + question ──► Qwen2.5-VL 7B (Ollama)    │
│                                    ▼                    │
│                             VQA answer                  │
└─────────────────────────────────────────────────────────┘
```

---

## Tech Stack

### Core models

| Role | Model | License | Size | Source |
|---|---|---|---|---|
| Multimodal embedding | `nomic-ai/nomic-embed-multimodal-3b` | Apache 2.0 | ~6 GB | HuggingFace |
| Sparse retrieval | `rank-bm25` | Apache 2.0 | — | pip |
| Cross-encoder reranker | `BAAI/bge-reranker-base` | MIT | ~278 MB | HuggingFace |
| Multimodal LLM (VQA) | `qwen2.5vl` | Apache 2.0 | ~6 GB (Q4) | Ollama |

### Infrastructure

| Role | Tool | License | Notes |
|---|---|---|---|
| Vector store | ChromaDB | Apache 2.0 | Local persistent, HNSW index |
| LLM serving | Ollama | MIT | Runs Qwen2.5-VL locally |
| RAG orchestration | LlamaIndex | MIT | Nomic + ChromaDB + Ollama integrations |
| BM25 indexing | rank-bm25 | Apache 2.0 | Pure Python, zero dependencies |
| Image processing | Pillow | HPND | Image loading and resizing |
| Evaluation | scikit-learn | BSD | Precision, recall, F1 metrics |
| UI (optional) | Streamlit | Apache 2.0 | Sprint 7 only |

### Development tools

| Tool | Purpose |
|---|---|
| Python 3.10+ | Runtime |
| VS Code | IDE |
| `uv` or `pip` | Package management |
| `python-dotenv` | Environment variable management |
| `pytest` | Unit and integration tests |
| `loguru` | Structured logging |

---

## Project Structure

```
multimodal-vqa-rag/
│
├── README.md
├── .env.example
├── requirements.txt
├── pyproject.toml
│
├── data/
│   ├── images/              # Raw input images (VG-150 / AUG / custom)
│   ├── captions/            # Auto-generated captions per image (for BM25)
│   └── chroma_db/           # ChromaDB persistent storage
│
├── src/
│   ├── __init__.py
│   │
│   ├── indexing/
│   │   ├── __init__.py
│   │   ├── embed.py         # Nomic Embed Multimodal — image → vector
│   │   ├── bm25_index.py    # BM25 sparse index builder
│   │   └── chroma_store.py  # ChromaDB collection management
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── dense.py         # Dense vector search via ChromaDB
│   │   ├── sparse.py        # BM25 keyword search
│   │   ├── rrf.py           # Reciprocal Rank Fusion
│   │   └── rerank.py        # BGE cross-encoder reranker
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   └── vqa.py           # Qwen2.5-VL via Ollama — prompt + answer
│   │
│   ├── pipeline.py          # End-to-end orchestrator
│   └── utils.py             # Shared helpers (image loading, logging)
│
├── eval/
│   ├── metrics.py           # Recall, F1, overall score
│   ├── run_eval.py          # Benchmark against VG-150 / AUG
│   └── results/             # Saved evaluation outputs
│
├── tests/
│   ├── test_embed.py
│   ├── test_retrieval.py
│   └── test_pipeline.py
│
└── app/
    └── ui.py                # Streamlit UI (Sprint 7)
```

---

## Sprints

---

### Sprint 0 — Environment Setup

**Goal:** Get every tool installed and verified before writing any pipeline code.

**Duration:** ~2 hours

#### Tasks

- [ ] Install Python 3.10+ and create a virtual environment
- [ ] Install Ollama and pull Qwen2.5-VL
- [ ] Install all Python dependencies
- [ ] Verify Ollama is running and the model responds
- [ ] Clone the project and set up VS Code with the Python extension

#### Commands

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

# 2. Install Ollama (visit https://ollama.com for installer)
#    Then pull the VQA model:
ollama pull qwen2.5vl

# 3. Verify Ollama is working
ollama run qwen2.5vl "describe what you can do"

# 4. Install Python dependencies
pip install torch torchvision pillow transformers
pip install chromadb rank-bm25 FlagEmbedding
pip install llama-index llama-index-llms-ollama
pip install python-dotenv loguru pytest streamlit
pip install scikit-learn tqdm

# 5. Create .env file
cp .env.example .env
```

#### Verify

```python
# run this quick sanity check
import chromadb
import ollama
from FlagEmbedding import FlagReranker
print("All imports OK")
```

#### Acceptance criteria

- `ollama run qwen2.5vl` responds in the terminal
- All Python imports succeed without errors
- VS Code Python extension shows the `.venv` interpreter

---

### Sprint 1 — Multimodal Embedding + Indexing

**Goal:** Build the indexing phase. Load images, embed them with Nomic, and store vectors in ChromaDB.

**Duration:** ~4 hours

**Files:** `src/indexing/embed.py`, `src/indexing/chroma_store.py`

#### Tasks

- [ ] Implement `embed.py` — load Nomic Embed Multimodal 3B from HuggingFace
- [ ] Write `embed_image(image_path)` that returns a 2048-dim numpy vector
- [ ] Write `embed_text(text)` for query-time embedding (same model, different mode)
- [ ] Implement `chroma_store.py` — create a ChromaDB collection with cosine distance
- [ ] Write `index_images(image_dir)` — batch embeds all images and upserts into Chroma
- [ ] Store image path and filename as metadata alongside each vector
- [ ] Write a test that indexes 10 images and queries for the nearest neighbour

#### Key code sketch

```python
# src/indexing/embed.py
from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch

MODEL_ID = "nomic-ai/nomic-embed-multimodal-3b"

def load_model():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID)
    return processor, model

def embed_image(image_path, processor, model) -> list[float]:
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    return outputs[0].tolist()  # 2048-dim vector
```

#### Acceptance criteria

- 100 images indexed in under 5 minutes on GPU
- ChromaDB collection persists to `data/chroma_db/` on disk
- Nearest-neighbour query returns the correct image for a trivial test case

---

### Sprint 2 — Hybrid Search + RRF Fusion

**Goal:** Add a BM25 sparse index on image captions and implement Reciprocal Rank Fusion to merge dense + sparse results.

**Duration:** ~3 hours

**Files:** `src/indexing/bm25_index.py`, `src/retrieval/dense.py`, `src/retrieval/sparse.py`, `src/retrieval/rrf.py`

#### Tasks

- [ ] Generate short captions for each image using Qwen2.5-VL (one-time, store as JSON)
- [ ] Build BM25 index over those captions using `rank-bm25`
- [ ] Implement `dense.py` — query ChromaDB, return ranked list of `(image_id, score)` tuples
- [ ] Implement `sparse.py` — query BM25, return ranked list of `(image_id, bm25_score)` tuples
- [ ] Implement `rrf.py` — fuse two ranked lists using `score = sum(1 / (k + rank))`
- [ ] Write unit tests for RRF with synthetic ranked lists
- [ ] Test that RRF correctly promotes images that appear in both lists

#### RRF implementation

```python
# src/retrieval/rrf.py
def rrf_fusion(dense_results: list, sparse_results: list, k: int = 60) -> list:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.
    Each list is [(image_id, score), ...] sorted by descending score.
    Returns merged list sorted by fused score.
    """
    scores = {}
    for rank, (img_id, _) in enumerate(dense_results):
        scores[img_id] = scores.get(img_id, 0) + 1 / (k + rank + 1)
    for rank, (img_id, _) in enumerate(sparse_results):
        scores[img_id] = scores.get(img_id, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

#### Acceptance criteria

- BM25 correctly surfaces images when the query contains exact object names
- RRF output consistently outperforms dense-only retrieval on manually checked examples
- Fusion takes under 10 ms per query

---

### Sprint 3 — Cross-Encoder Re-ranking

**Goal:** Apply BGE-reranker-base to the fused top-N results to produce a final top-4 shortlist.

**Duration:** ~2 hours

**Files:** `src/retrieval/rerank.py`

#### Tasks

- [ ] Load `BAAI/bge-reranker-base` using `FlagEmbedding`
- [ ] Implement `rerank(query, candidates)` — takes query text and list of (image_id, caption) pairs
- [ ] Score each (query, caption) pair jointly using the cross-encoder
- [ ] Return top-4 image IDs sorted by reranker score
- [ ] Write a test showing that reranking changes the order for ambiguous queries

#### Key code sketch

```python
# src/retrieval/rerank.py
from FlagEmbedding import FlagReranker

reranker = FlagReranker("BAAI/bge-reranker-base", use_fp16=True)

def rerank(query: str, candidates: list[tuple[str, str]], top_k: int = 4) -> list[str]:
    """
    candidates: [(image_id, caption), ...]
    Returns top_k image_ids sorted by cross-encoder score.
    """
    pairs = [[query, caption] for _, caption in candidates]
    scores = reranker.compute_score(pairs)
    ranked = sorted(zip([img_id for img_id, _ in candidates], scores),
                    key=lambda x: x[1], reverse=True)
    return [img_id for img_id, _ in ranked[:top_k]]
```

#### Acceptance criteria

- Reranker runs in under 500 ms for top-20 candidates on CPU
- Order changes meaningfully versus RRF alone on at least 30% of test queries
- No CUDA errors when running on CPU fallback

---

### Sprint 4 — VQA Generation via Ollama

**Goal:** Pass the top-4 retrieved images plus the user question to Qwen2.5-VL and return a grounded answer.

**Duration:** ~3 hours

**Files:** `src/generation/vqa.py`

#### Tasks

- [ ] Connect to Ollama's local API using the `ollama` Python SDK
- [ ] Implement `generate_answer(question, image_paths)` — sends images + question to Qwen2.5-VL
- [ ] Build the prompt template: retrieved image context + user question
- [ ] Handle Ollama errors gracefully (model not loaded, timeout)
- [ ] Test with a sample image and question to verify the model responds correctly
- [ ] Tune `temperature`, `top_p`, and `max_tokens` for VQA quality

#### Key code sketch

```python
# src/generation/vqa.py
import ollama

SYSTEM_PROMPT = """You are an expert visual analyst. 
You will be shown retrieved images and asked a question about them.
Answer accurately based only on what you can observe in the images.
Focus on: object counts, spatial locations, and relationships between objects."""

def generate_answer(question: str, image_paths: list[str]) -> str:
    response = ollama.chat(
        model="qwen2.5vl",
        messages=[{
            "role": "user",
            "content": question,
            "images": image_paths
        }],
        options={"temperature": 0.1, "top_p": 0.9}
    )
    return response["message"]["content"]
```

#### Acceptance criteria

- Model returns a coherent answer for any input image + question pair
- Answers mention specific object counts and locations from the image
- Graceful error message if Ollama is not running

---

### Sprint 5 — Pipeline Orchestration + CLI

**Goal:** Wire all four components into a single end-to-end pipeline with a clean CLI interface.

**Duration:** ~3 hours

**Files:** `src/pipeline.py`, `src/utils.py`

#### Tasks

- [ ] Implement `VQAPipeline` class that wires embed → retrieve → rerank → generate
- [ ] Expose two public methods: `index(image_dir)` and `query(question, image_path=None)`
- [ ] Add a `--index` CLI flag to re-index an image directory
- [ ] Add a `--query` CLI flag to ask a question
- [ ] Add `--image` flag to optionally include a query image alongside the text question
- [ ] Add structured logging at each pipeline stage with timing information
- [ ] Write an integration test that runs the full pipeline on 5 images

#### CLI usage

```bash
# Index all images in a directory
python -m src.pipeline --index --image-dir data/images/

# Ask a question
python -m src.pipeline --query "How many people are in the top-left corner?"

# Ask a question with a query image
python -m src.pipeline --query "What objects are similar to this?" --image data/images/test.jpg
```

#### Acceptance criteria

- Full pipeline runs end-to-end without errors on at least 20 images
- Total latency from query to answer is under 15 s on GPU
- CLI prints a human-readable answer with timing breakdown per stage

---

### Sprint 6 — Evaluation + Benchmarking

**Goal:** Evaluate the revised pipeline against the original paper's metrics on VG-150 and/or AUG dataset.

**Duration:** ~4 hours

**Files:** `eval/metrics.py`, `eval/run_eval.py`

#### Tasks

- [ ] Implement recall, precision, F1 calculation for: category, quantity, location, relationship
- [ ] Implement the paper's "overall score" (ratio of classes with recall ≥ 0.55)
- [ ] Download and prepare the VG-150 test split (100 images from the test set)
- [ ] Run the pipeline on all 100 images and save answers to `eval/results/`
- [ ] Parse answers to extract predicted categories, quantities, locations, relationships
- [ ] Compare F1 and recall scores against the paper's Table I and Table II baselines
- [ ] Generate a markdown comparison report

#### Metrics to track

```
Category recall / F1
Quantity recall / F1
Location recall / F1
Relationship recall / F1
Overall score (recall >= 0.55 threshold)
```

#### Acceptance criteria

- Category recall on VG-150 ≥ 0.55 (paper's original method scored 0.55)
- Location recall improved over original (target ≥ 0.18 vs paper's 0.13)
- Evaluation report saved as `eval/results/comparison_report.md`

---

### Sprint 7 — Streamlit UI (Optional)

**Goal:** Build a simple browser-based UI for demo purposes.

**Duration:** ~3 hours

**Files:** `app/ui.py`

#### Tasks

- [ ] Create a Streamlit app with an image upload widget
- [ ] Add a text input for the user's question
- [ ] Display the top-4 retrieved images as a grid
- [ ] Show the generated VQA answer prominently
- [ ] Add a sidebar with pipeline settings (top-k, model temperature)
- [ ] Show per-stage latency breakdown in an expandable section

#### Run the UI

```bash
streamlit run app/ui.py
```

#### Acceptance criteria

- UI loads at `http://localhost:8501`
- User can upload an image, type a question, and get an answer in under 20 s
- Retrieved images are displayed alongside the answer

---

## Environment Variables

Create a `.env` file in the project root:

```env
# Ollama settings
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen2.5vl

# Embedding model
EMBED_MODEL_ID=nomic-ai/nomic-embed-multimodal-3b
EMBED_DEVICE=cuda          # or cpu

# Reranker
RERANKER_MODEL_ID=BAAI/bge-reranker-base
RERANKER_TOP_K=4

# ChromaDB
CHROMA_PERSIST_DIR=./data/chroma_db
CHROMA_COLLECTION=vqa_images

# BM25
BM25_CAPTIONS_FILE=./data/captions/captions.json

# Retrieval
DENSE_TOP_N=20             # candidates before RRF
SPARSE_TOP_N=20
RRF_K=60
```

---

## How to Run

### Full pipeline from scratch

```bash
# Step 1 — activate venv
source .venv/bin/activate

# Step 2 — start Ollama (if not already running)
ollama serve &

# Step 3 — index your image directory
python -m src.pipeline --index --image-dir data/images/

# Step 4 — ask a question
python -m src.pipeline --query "How many cars are in the center of the image?"
```

### Run evaluation

```bash
python eval/run_eval.py --dataset vg150 --split test
```

### Run tests

```bash
pytest tests/ -v
```

---

## Datasets

### VG-150 (Visual Genome subset)

- 108,077 images total; we use the 150 most frequent object categories
- Download: [https://visualgenome.org/api/v0/api_home.html](https://visualgenome.org/api/v0/api_home.html)
- Use the test split of 100 images for evaluation (same as the paper)

### AUG (Aerial Urban Scenes)

- 400 aerial images; 300 train / 100 test
- 77 object categories, 63 relationship types
- Paper: [https://arxiv.org/abs/2404.07788](https://arxiv.org/abs/2404.07788)

### Custom images

Place any JPEG/PNG images in `data/images/` and run `--index`. The pipeline works on any image corpus.

---

## Cost

| Component | Cost |
|---|---|
| Nomic Embed Multimodal 3B | $0 — Apache 2.0, runs locally |
| ChromaDB | $0 — Apache 2.0, runs locally |
| rank-bm25 | $0 — Apache 2.0, pip package |
| BGE-reranker-base | $0 — MIT, runs locally |
| Qwen2.5-VL 7B via Ollama | $0 — Apache 2.0, runs locally |
| Google Colab T4 (if no GPU) | $0 — free tier |
| **Total** | **$0** |

---

## References

- Xue et al., *"Enhanced Multimodal RAG-LLM for Accurate Visual Question Answering"*, arXiv:2412.20927, Dec 2024
- Nomic AI, *"Nomic Embed Multimodal"*, 2025 — [https://nomic.ai/news/nomic-embed-multimodal](https://nomic.ai/news/nomic-embed-multimodal)
- BAAI, *"BGE-M3 and BGE Reranker"*, 2024 — [https://github.com/FlagOpen/FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)
- Cormack et al., *"Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"*, SIGIR 2009
- Ollama — [https://ollama.com](https://ollama.com)
- ChromaDB — [https://www.trychroma.com](https://www.trychroma.com)