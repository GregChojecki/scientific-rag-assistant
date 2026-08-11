from src.scientific_rag.vector_store import create_chroma_client


def test_create_chroma_client():
    client = create_chroma_client()

    assert client is not None