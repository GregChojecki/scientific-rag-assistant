from sentence_transformers import SentenceTransformer


def load_embedding_model(
    model_name: str = "all-MiniLM-L6-v2",
) -> SentenceTransformer:
    """
    Load and return a sentence-transformers embedding model.
    """
    return SentenceTransformer(model_name)

def embed_texts(
    texts: list[str],
    model: SentenceTransformer,
):
    """
    Convert a list of text strings into embedding vectors.
    """
    return model.encode(texts)