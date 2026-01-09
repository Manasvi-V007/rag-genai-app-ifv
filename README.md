# URL-Based RAG GenAI App (Gemini 2.5 Flash)

## Overview
This project implements a Retrieval-Augmented Generation (RAG) pipeline
using the latest google.genai SDK. Web content is dynamically retrieved
from a user-provided URL, embedded using Gemini embedding models, and
stored in a FAISS vector index. Relevant content is retrieved at query
time and passed to Gemini 2.5 Flash for grounded answer generation.

## Architecture
URL → Loader → Chunking → Embeddings (google.genai)
→ FAISS → Retrieval → Gemini 2.5 Flash

## Run Instructions
1. Install dependencies:
   pip install -r requirements.txt

2. Create a `.env` file and add your Gemini API key

3. Run:
   streamlit run app.py
