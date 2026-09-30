# Multi-PDF RAG Assistant

A Streamlit-based Retrieval-Augmented Generation assistant for multiple PDF documents.

## Architecture

PDFs -> PyMuPDF -> Recursive Chunking -> Sentence Transformers -> ChromaDB -> Similarity Search -> Threshold -> Cohere Rerank -> Cohere LLM -> Answer + Sources

## Hybrid design

Embeddings run locally with Sentence Transformers, so PDF ingestion does not consume Cohere embedding API tokens.

Cohere is used for reranking and final answer generation.

## Features

- Multiple PDF upload
- Search one or multiple PDFs
- Semantic retrieval with ChromaDB
- Top K control
- Similarity distance threshold
- Optional Cohere Rerank
- Conversation history
- PDF filename and page sources
- Grounded-answer prompt
- Unknown-answer handling

## Chunking

chunk_size = 500
chunk_overlap = 100

The smaller chunk size keeps retrieved passages focused. The overlap preserves context between neighboring chunks.

## Secrets

For local development, set COHERE_API_KEY in the environment.

For Streamlit Cloud, add:

COHERE_API_KEY = "your_key_here"

Never commit API keys to GitHub.

## Run

Install dependencies from requirements.txt, then run:

streamlit run app.py

## Evaluation

Use at least three PDFs and test:
1. A question answered only by PDF 1.
2. A question answered only by PDF 2.
3. A question answered only by PDF 3.
4. A question requiring two PDFs.
5. A question absent from all PDFs.

Record screenshots showing answers and PDF/page sources.
