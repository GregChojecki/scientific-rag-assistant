from src.scientific_rag.vector_store import (
    add_documents,
    create_chroma_client,
)


def test_create_chroma_client():
    client = create_chroma_client()

    assert client is not None


def test_add_documents():
    client = create_chroma_client()

    documents = ["machine learning", "scientific retrieval"]
    embeddings = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    collection = add_documents(
        client=client,
        collection_name="test_collection",
        documents=documents,
        embeddings=embeddings,
    )

    assert collection.count() == 2