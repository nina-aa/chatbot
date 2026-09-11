# Multi-PDF RAG System

A command-line RAG (Retrieval-Augmented Generation) tool that answers questions over two PDF documents at once, using OpenAI embeddings + `gpt-3.5-turbo` and a local FAISS vector store.

## Features

- Ingests **two PDFs** per run and lets you query across both
- Caches embeddings to disk (per-PDF and combined) so repeat runs cost **zero** tokens
- Shows retrieved source chunks with similarity scores and page numbers
- Tracks OpenAI token usage (input/output/total) per query
- Confirms estimated embedding cost before spending any tokens

## Prerequisites

- Python 3.8+
- An [OpenAI API key](https://platform.openai.com/api-keys)

## Setup

```bash
# 1. (Optional) Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your API key

## Usage

```bash
python rag_system7.py --pdf1 HMW90.pdf --pdf2 WAA.pdf
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--pdf1` | *(required)* | Path to the first PDF |
| `--pdf2` | *(required)* | Path to the second PDF |
| `--chunk-size` | `500` | Characters per text chunk |
| `--chunk-overlap` | `100` | Overlap between chunks |
| `--rebuild` | off | Force-rebuild the combined vector store (reuses individual PDF vector stores if they already exist) |

On first run for a given PDF pair, you'll see an embedding cost estimate and a prompt to confirm before any API calls are made. Vector stores are then saved locally (e.g. `HMW90_WAA_vectorstore_chunk500/`) and reused automatically on future runs with the same PDFs and chunk size.

Once ready, ask questions interactively. Type `quit`, `exit`, or `q` to stop.

## Sample PDFs

This repo includes a few sample PDFs (`HMW90.pdf`, `HPP.pdf`, `WAA.pdf`, `vai.pdf`) as public source material to try the system against. Swap in your own PDFs by passing different paths to `--pdf1`/`--pdf2`.

## Future Directions

**A. Improving Retrieval Quality**
- Reranking models (score chunks after retrieval for relevance)
- Hybrid search (combine keyword + semantic search)
- Query reformulation (rephrase user questions for better matching)
- Conversation history, so follow-up questions work better

**B. Query Routing**
- Route queries to the right manual automatically
- Prevent irrelevant queries from consuming tokens
- Improve GUI usability

**C. Understanding Trade-offs**
- Chunk size vs. context preservation
- Number of chunks vs. token cost
- Speed vs. accuracy

**D. Metadata and Filtering**
- Add page numbers and section headers to chunks
- Filter by metadata before similarity search

