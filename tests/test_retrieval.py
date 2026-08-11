from src.scientific_rag.retrieval import retrieve_documents
from src.scientific_rag.vector_store import add_documents, create_chroma_client


def test_retrieve_documents_returns_most_similar_document():
    client = create_chroma_client()

    documents = [
        "machine learning",
        "scientific retrieval",
    ]

    embeddings = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    collection = add_documents(
        client=client,
        collection_name="retrieval_test",
        documents=documents,
        embeddings=embeddings,
    )

    result = retrieve_documents(
        collection=collection,
        query_embedding=[1.0, 0.0],
        n_results=1,
    )

    assert result == ["machine learning"]