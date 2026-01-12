import os
import hashlib
import json
from dotenv import load_dotenv

import numpy as np
import faiss
from sklearn.metrics.pairwise import cosine_similarity

from openai import AzureOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load env
load_dotenv()

# Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
EMBEDDING_MODEL = "text-embedding-3-large"

# ---------------- Embedding helpers ----------------

def get_text_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def get_embeddings(texts: list[str]) -> np.ndarray:
    vectors = []
    for text in texts:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        vectors.append(response.data[0].embedding)
    return np.array(vectors).astype("float32")


# ---------------- RAG Pipeline ----------------

def answer_from_url(url: str, question: str, debug: bool = True) -> str:
    try:
        # Load webpage
        loader = WebBaseLoader(url)
        documents = loader.load()
        if not documents:
            return "Error: Unable to load content from URL."

        # Chunk text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(documents)
        texts = [doc.page_content for doc in chunks]

        if debug:
            print(f"[DEBUG] Chunks created: {len(texts)}")

        # Generate embeddings
        embeddings = get_embeddings(texts)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        query_embedding = get_embeddings([question])
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

        # Similarity search
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        top_k = min(7, len(texts))
        top_indices = np.argsort(similarities)[::-1][:top_k]

        relevant_text = "\n\n".join(texts[i] for i in top_indices)

        if debug:
            print(f"[DEBUG] Retrieved context length: {len(relevant_text)}")

        # Prompt
        prompt = f"""
You are a factual assistant.
Answer ONLY using the context below.
If the answer is not present, say:
"Information not found in the provided source."

Context:
{relevant_text}

Question:
{question}

Answer:
"""

        # Azure GPT-4o generation
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error occurred: {str(e)}"
