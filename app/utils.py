import os
import json
import numpy as np
import faiss
import fitz  # PyMuPDF
from django.conf import settings
from sentence_transformers import SentenceTransformer

# Load the model once globally for performance
model = SentenceTransformer('all-MiniLM-L6-v2')

# Simple chunker: splits long text into chunks with overlap
def chunk_text(text, max_chars=1500, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start+max_chars]
        chunks.append(chunk)
        start += max_chars - overlap
    return chunks

# Extract text per page from PDF
def extract_pages_text(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        text = doc[i].get_text()
        pages.append({'page': i+1, 'text': text})
    return pages

# Load existing FAISS index or None
def load_faiss(index_path):
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
    else:
        index = None
    return index

# Add vectors and metadata to FAISS index
def upsert_vectors(vectors, metadatas, index_path):
    dim = len(vectors[0])
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
    else:
        index = faiss.IndexFlatL2(dim)
    arr = np.array(vectors, dtype='float32')
    index.add(arr)
    faiss.write_index(index, index_path)
    # Store metadata alongside index
    meta_path = index_path + '.meta.json'
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            existing = json.load(f)
    else:
        existing = []
    existing.extend(metadatas)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(existing, f)

# Search FAISS index for nearest neighbors
def search_index(query_vector, index_path, top_k=3):
    index = faiss.read_index(index_path)
    q = np.array([query_vector], dtype='float32')
    D, I = index.search(q, top_k)
    meta_path = index_path + '.meta.json'
    with open(meta_path, 'r', encoding='utf-8') as f:
        metas = json.load(f)
    results = []
    for idx in I[0]:
        if idx < len(metas):
            results.append(metas[idx])
    return results

# Get embedding for text using SentenceTransformer
def get_embedding(text):
    embedding = model.encode(text)
    return embedding.tolist()

# Simple local "LLM" substitute that summarizes contexts for answering
def ask_llm(question, contexts):
    # Just concatenate context texts for a naive summary answer
    context_text = " ".join(c['text'] for c in contexts)
    # Return a simple summary snippet and all context pages
    answer = f"Based on the document context: {context_text[:500]}..."
    pages = [c['page'] for c in contexts]
    return answer, pages
