# URL-Based RAG GenAI App (Azure OpenAI – GPT-4o)

## Overview
This project implements a Retrieval-Augmented Generation (RAG) pipeline
using Azure OpenAI services. Web content is dynamically retrieved from a
user-provided URL, embedded using Azure OpenAI embedding models, and
relevant context is retrieved to generate grounded answers using GPT-4o.

The application is built using Streamlit and deployed on Streamlit Cloud.

## Architecture
URL → Loader → Chunking → Embeddings (Azure OpenAI)
→ Similarity Retrieval → GPT-4o → Answer

## Run Instructions
1. Install dependencies:
   pip install -r requirements.txt

2. Create a `.env` file and add your Azure OpenAI credentials

3. Run:
   streamlit run app.py
