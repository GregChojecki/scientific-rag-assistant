import chromadb


def create_chroma_client() -> chromadb.ClientAPI:
    """
    Create and return an in-memory ChromaDB client.
    """
    return chromadb.Client()