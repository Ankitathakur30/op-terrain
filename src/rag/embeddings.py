import os
os.environ["HF_HUB_DISABLE_XET"]="1"

from sentence_transformers import SentenceTransformer
_model=None
def get_model():
    global _model
    if _model is None:
        _model=SentenceTransformer("all-MiniLM-L6-v2")
        return _model

def embed_texts(texts):
    model=get_model()
    embeddings=model.encode(texts,convert_to_numpy=True,show_progress_bar=False)
    return embeddings