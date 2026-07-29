import faiss
import pickle
import numpy as np
from src.rag.embeddings import embed_texts

def retrieve(query,k=3, index_path="data/processed/faiss.index", texts_path="data/processed/chunk_texts.pkl"):
    index = faiss.read_index(index_path)
    with open(texts_path, "rb") as f:
        chunks = pickle.load(f)

    query_vec = embed_texts([query])
    faiss.normalize_L2(query_vec)

    scores, indices = index.search(query_vec, k)
    results = [(chunks[i], float(scores[0][rank])) for rank, i in enumerate(indices[0])]
    return results

if __name__ == "__main__":
    results = retrieve("how should be the foot steps for movement on snow and how to make movement easier")
    for chunk, score in results:
        print(f"\n[score: {score:.3f}]\n{chunk[:200]}...")