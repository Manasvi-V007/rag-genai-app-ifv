import os
from dotenv import load_dotenv
import hashlib
import json

import numpy as np
import faiss
from google import genai
from sklearn.metrics.pairwise import cosine_similarity

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Initialize Google GenAI client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Embedding cache file
CACHE_FILE = "embedding_cache.json"

def load_embedding_cache():
    """Load embedding cache from file"""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_embedding_cache(cache):
    """Save embedding cache to file"""
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)

def get_text_hash(text: str) -> str:
    """Generate hash for text"""
    return hashlib.md5(text.encode()).hexdigest()

def get_embeddings(texts: list[str], use_cache: bool = False) -> np.ndarray:
    """
    Generate embeddings using google.genai (latest SDK) with caching
    """
    cache = load_embedding_cache() if use_cache else {}
    embeddings = []
    texts_to_embed = []
    indices_to_embed = []
    
    # Check which texts need embedding
    for i, text in enumerate(texts):
        text_hash = get_text_hash(text)
        if text_hash in cache:
            embeddings.append(cache[text_hash])
        else:
            texts_to_embed.append(text)
            indices_to_embed.append(i)
    
    # Call API only for texts not in cache
    if texts_to_embed:
        for text in texts_to_embed:
            response = client.models.embed_content(
                model="models/embedding-001",
                contents=text
            )
            text_hash = get_text_hash(text)
            embedding = response.embeddings[0].values
            
            # Validate embedding
            if embedding is None or len(embedding) == 0:
                print(f"[ERROR] Empty embedding received for text: {text[:50]}...")
                continue
                
            cache[text_hash] = embedding
            embeddings.append(embedding)
        
        # Save updated cache only if use_cache is enabled
        if use_cache:
            save_embedding_cache(cache)
    
    # Sort embeddings back to original order
    final_embeddings = [None] * len(texts)
    cache_idx = 0
    embed_idx = 0
    for i, text in enumerate(texts):
        text_hash = get_text_hash(text)
        if text_hash in cache:
            if i in indices_to_embed:
                final_embeddings[i] = embeddings[len(texts) - len(texts_to_embed) + embed_idx]
                embed_idx += 1
            else:
                final_embeddings[i] = cache[text_hash]

    # Validate all embeddings are present
    if None in final_embeddings:
        print("[ERROR] Some embeddings are missing!")
        return None
    
    return np.array(final_embeddings).astype("float32")


def answer_from_url(url: str, question: str, debug: bool = True) -> str:
    """
    RAG Pipeline:
    URL → Load → Chunk → Embed (google.genai)
        → Cosine Similarity → Retrieve → Gemini 2.5 Flash
    """

    try:
        #  Load webpage content
        loader = WebBaseLoader(url)
        documents = loader.load()
        if debug:
            print(f"[DEBUG] Loaded {len(documents)} document(s)")
        
        if not documents:
            return "Error: Could not load content from the URL. Ensure the URL is accessible."

        #  Split content into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Increased for better context
            chunk_overlap=200
        )
        chunks = splitter.split_documents(documents)
        if debug:
            print(f"[DEBUG] Created {len(chunks)} chunks")
            if chunks:
                print(f"[DEBUG] First chunk: {chunks[0].page_content[:150]}...")

        texts = [doc.page_content for doc in chunks]
        
        if not texts:
            return "Error: No content chunks extracted from URL."

        #  Create embeddings (google.genai – NOT deprecated)
        embeddings = get_embeddings(texts, use_cache=False)
        if embeddings is None:
            return "Error: Failed to generate embeddings."
        if debug:
            print(f"[DEBUG] Embedding shape: {embeddings.shape}")

        #  Normalize embeddings for cosine similarity
        embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        #  Embed the query
        query_embedding = get_embeddings([question], use_cache=False)
        if query_embedding is None:
            return "Error: Failed to embed question."
        
        # Normalize query embedding
        query_normalized = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        if debug:
            print(f"[DEBUG] Query embedding shape: {query_normalized.shape}")

        #  Retrieve top-k relevant chunks using cosine similarity
        k = min(7, len(texts))  # Retrieve up to 7 chunks
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_normalized, embeddings_normalized)[0]
        
        # Get top-k indices sorted by similarity (highest first)
        top_k_indices = np.argsort(similarities)[::-1][:k]
        top_k_scores = similarities[top_k_indices]
        
        if debug:
            print(f"[DEBUG] Top-k indices: {top_k_indices}, scores: {top_k_scores}")
            for i, idx in enumerate(top_k_indices):
                print(f"[DEBUG] Chunk {idx} (similarity: {top_k_scores[i]:.4f}): {texts[idx][:150]}...")

        # Filter by minimum similarity threshold
        min_similarity = 0.3  # Adjust if needed
        relevant_indices = top_k_indices[top_k_scores >= min_similarity]
        
        if len(relevant_indices) == 0:
            # If no chunks meet threshold, still use top chunks but with warning
            relevant_indices = top_k_indices[:k]
            if debug:
                print(f"[DEBUG] No chunks met similarity threshold {min_similarity}, using top {k}")

        retrieved_context = "\n\n".join(
            texts[i] for i in relevant_indices
        )
        
        if debug:
            print(f"[DEBUG] Retrieved context length: {len(retrieved_context)}")
        
        if not retrieved_context or len(retrieved_context.strip()) < 50:
            return f"Error: Could not retrieve relevant content."

        #  Grounded prompt
        prompt = f"""You are a factual assistant. Answer the question ONLY using the context below.
If the answer is not in the context, say "Information not found in the provided source."

Context:
{retrieved_context}

Question: {question}

Answer:"""

        #  Gemini 2.5 Flash generation
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error occurred: {str(e)}"
