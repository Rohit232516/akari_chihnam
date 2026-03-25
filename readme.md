# 🧠 Multimodal VQA RAG Pipeline

> A fully free, locally-run Visual Question Answering system built with multimodal embeddings, hybrid search, RRF fusion, cross-encoder re-ranking, and Gemini 2.5 Flash. Inspired by the paper *"Enhanced Multimodal RAG-LLM for Accurate Visual Question Answering"* (Xue et al., Dec 2024).

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Gemini](https://img.shields.io/badge/Gemini-2.5%20Flash-orange)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-green)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📸 Demo

Upload any image and ask questions about it — the assistant answers using Gemini 2.5 Flash with full multimodal understanding.

```
User: "How many people are in this image and where are they located?"
Assistant: "There are 3 people. One is standing in the top-left corner near
            the building entrance, and two are seated in the center of the frame."
```

---

## 🏗️ Architecture

The system has two operating modes:

### Mode 1 — Direct VQA (user uploads image)
```
User image + Question → Gemini 2.5 Flash → Answer
```

### Mode 2 — RAG VQA (query over indexed corpus)
```
Question
  ├── Dense retrieval  → Gemini Embedding 2 → cosine search ChromaDB
  └── Sparse retrieval → BM25 keyword search over captions
           ↓
     RRF fusion (k=60)
           ↓
     BGE cross-encoder reranker
           ↓
     Top-4 images + Question → Gemini 2.5 Flash → Answer
```

### Why hybrid search?

Dense vectors capture semantic meaning but dilute exact specifics. BM25 catches exact keywords — object names, counts, spatial terms — that dense vectors average away. RRF merges both ranked lists without needing score normalisation.

---

## 🔧 Tech Stack

| Component | Tool | Notes |
|---|---|---|
| Multimodal embedding | `gemini-embedding-2-preview` | Images + text in same vector space |
| Vector store | ChromaDB | Local persistent, HNSW index |
| Sparse retrieval | rank-bm25 | Keyword search over captions |
| Fusion | RRF (k=60) | Merges dense + sparse lists |
| Re-ranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder, runs locally |
| Caption generation | moondream via Ollama | Generates BM25 captions |
| VQA generation | `gemini-2.5-flash` | Multimodal LLM |
| UI | Streamlit | ChatGPT-style interface |
| Language | Python 3.10+ | |

---

## 📁 Project Structure

```
multimodal-vqa-rag/
│
├── app/
│   └── app.py                  # Streamlit UI
│
├── src/
│   ├── pipeline.py             # End-to-end orchestrator + CLI
│   │
│   ├── indexing/
│   │   ├── embed.py            # Gemini Embedding 2 — image + text → vector
│   │   ├── chroma_store.py     # ChromaDB vector store management
│   │   ├── bm25_index.py       # BM25 sparse index over image captions
│   │   └── indexer.py          # Indexing pipeline orchestrator
│   │
│   ├── retrieval/
│   │   ├── retriever.py        # Hybrid retriever (dense + sparse + RRF)
│   │   ├── dense.py            # Dense vector search
│   │   ├── sparse.py           # BM25 keyword search
│   │   ├── rrf.py              # Reciprocal Rank Fusion
│   │   └── rerank.py           # BGE cross-encoder reranker
│   │
│   └── generation/
│       └── vqa.py              # Gemini 2.5 Flash VQA generation
│
├── data/
│   ├── images/                 # Input image corpus
│   ├── captions/               # Auto-generated BM25 captions (JSON)
│   └── chroma_db/              # Persistent ChromaDB vector store
│
├── tests/
│   └── test_sprint1.py         # Unit + integration tests
│
├── eval/
│   └── metrics.py              # Recall, F1, overall score evaluation
│
├── .env.example                # Environment variable template
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- macOS / Linux
- A free Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
- Ollama (for caption generation during indexing)

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/multimodal-vqa-rag.git
cd multimodal-vqa-rag
```

### 2. Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Install Ollama and pull moondream

```bash
brew install ollama
ollama pull moondream
ollama serve &
```

### 4. Set up environment variables

```bash
cp .env.example .env
```

Edit `.env` and add your Gemini API key:

```env
GEMINI_API_KEY=your_gemini_api_key_here
EMBED_MODEL_ID=gemini-embedding-2-preview
VQA_MODEL=gemini-2.5-flash
OLLAMA_MODEL=moondream
```

### 5. Run the Streamlit app

```bash
streamlit run app/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 💬 Usage

### Direct VQA (recommended for single images)

1. Open the app in your browser
2. Upload an image in the **sidebar**
3. Type your question in the chat input
4. Get an answer powered by Gemini 2.5 Flash

### RAG mode (for querying over a corpus)

First index your image directory:

```bash
python3 -m src.pipeline --index --image-dir data/images/
```

Then ask questions without uploading — the system retrieves relevant images from the corpus automatically.

### CLI usage

```bash
# index a directory of images
python3 -m src.pipeline --index --image-dir data/images/

# ask a question (RAG mode)
python3 -m src.pipeline --query "How many cars are in the image?"

# force re-index
python3 -m src.pipeline --index --image-dir data/images/ --reindex
```

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | — | **Required.** Get free at [AI Studio](https://aistudio.google.com/app/apikey) |
| `EMBED_MODEL_ID` | `gemini-embedding-2-preview` | Gemini multimodal embedding model |
| `VQA_MODEL` | `gemini-2.5-flash` | Gemini model for answer generation |
| `OLLAMA_MODEL` | `moondream` | Local model for BM25 caption generation |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | ChromaDB storage path |
| `CHROMA_COLLECTION_NAME` | `vqa_images` | ChromaDB collection name |
| `BM25_CAPTIONS_FILE` | `./data/captions/captions.json` | Caption file path |
| `DENSE_TOP_N` | `20` | Dense retrieval candidate count |
| `SPARSE_TOP_N` | `20` | Sparse retrieval candidate count |
| `RRF_K` | `60` | RRF fusion constant |
| `RERANKER_TOP_K` | `4` | Final images passed to LLM |

---

## 🔍 How Embedding Works

### The key insight

Gemini Embedding 2 maps **images and text into the same vector space**. This means:

```
image.jpg  →  Gemini Embed  →  [0.12, -0.87, ...]  →  ChromaDB
"red car"  →  Gemini Embed  →  [0.11, -0.85, ...]  →  cosine similarity → match!
```

No text description of the image is ever needed for dense retrieval.

### Why BM25 still matters

Dense vectors average out specific details. "3 red cars" becomes a general "cars" concept. BM25 catches exact keywords — counts, colours, positions — that dense vectors lose. The RRF fusion combines both strengths.

---

## 📊 Improvements Over the Original Paper

| Dimension | Original paper | This project |
|---|---|---|
| Image embedding | Faster-RCNN + PENET → text chunks | Direct image embedding (Gemini) |
| Embedding model | Text2Vec Multilingual (text-only) | Gemini Embedding 2 (image + text) |
| Retrieval | Cosine similarity only | Hybrid search + RRF + rerank |
| Vector store | In-memory (no persistence) | ChromaDB (persistent) |
| LLM | Qwen-2-72B-Instruct | Gemini 2.5 Flash |
| Preprocessing | Faster-RCNN + PENET + GloVe | None — direct embedding |
| Cost | Local GPU required | Free (Gemini free tier) |

---

## 🧪 Running Tests

```bash
python3 -m pytest tests/test_sprint1.py -v
```

Tests cover: embedding model loading, ChromaDB upsert/query, BM25 build/search, and full dense leg integration.

---

## 📚 References

- Xue et al., *"Enhanced Multimodal RAG-LLM for Accurate Visual Question Answering"*, arXiv:2412.20927, Dec 2024
- Google, *"Gemini Embedding 2"*, March 2026
- BAAI, *"BGE Reranker"*, 2024
- Cormack et al., *"Reciprocal Rank Fusion"*, SIGIR 2009
- [ChromaDB](https://www.trychroma.com) · [Ollama](https://ollama.com) · [Streamlit](https://streamlit.io)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙋 Author

Built by **Narayanam Rohit** · [GitHub](https://github.com/yourusername)