# Ask My Docs - Production RAG Application

A domain-specific Retrieval-Augmented Generation system that allows users to upload documents and ask questions with citation-backed answers.

## Features

- PDF/TXT document upload
- Text chunking
- Hybrid retrieval using BM25 + vector search
- Cross-encoder reranking
- Citation-enforced answer generation
- Hallucination-safe fallback
- Streamlit web interface
- Free deployment ready

## Tech Stack

- Python
- Streamlit
- Sentence Transformers
- BM25
- Cross Encoder Reranker
- Groq LLM API

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py