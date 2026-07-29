import os
import faiss
import numpy as np
import pickle
from src.rag.embeddings import embed_texts

def chunk_text(text,chunk_size=300,overlap=50):
    words=text.split()
    chunks=[]
    start=0
    while start<len(words):
        end=start+chunk_size
        chunk=" ".join(words[start:end])
        chunks.append(chunk)
        start+=chunk_size-overlap
    return chunks

def build_vectorstore(doc_dir="knowledge_base/doctrine_excerpts",index_path="data/processed/faiss.index",texts_path="data/processed/chunk_texts.pkl"):
    all_chunks=[]
    for filename in os.listdir(doc_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(doc_dir,filename),"r",encoding="utf-8") as f:
                text=f.read()
                all_chunks.extend(chunk_text(text))

    embeddings=embed_texts(all_chunks)
    dimension=embeddings.shape[1]
    index=faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    os.makedirs(os.path.dirname(index_path),exist_ok=True)
    faiss.write_index(index,index_path)
    with open(texts_path,"wb") as f:
        pickle.dump(all_chunks,f)
    
    print(f"Indexed {len(all_chunks)} chunks from {doc_dir}")
    return index,all_chunks
if __name__=="__main__":
    build_vectorstore()