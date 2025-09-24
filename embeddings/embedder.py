from sentence_transformers import SentenceTransformer

def embed_chunks(chunks, model):
    """
    Returns embeddings for the provided list of chunks using the supplied SentenceTransformer model.
    """
    embeddings = model.encode(
        chunks, 
        convert_to_numpy=True, 
        normalize_embeddings=True
    )
    return embeddings