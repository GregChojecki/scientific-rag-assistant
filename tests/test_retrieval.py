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


from src.scientific_rag.rag_pipeline import retrieve_context
from src.scientific_rag.embeddings import load_embedding_model


def test_retrieve_context_returns_relevant_documents():
    client = create_chroma_client()

    documents = [
        "machine learning",
        "scientific retrieval",
    ]

    model = load_embedding_model()
    embeddings = model.encode(documents)

    collection = add_documents(
        client=client,
        collection_name="context_test",
        documents=documents,
        embeddings=embeddings,
    )

    results = retrieve_context(
        question="scientific retrieval",
        collection=collection,
        model=model,
        n_results=1,
    )

    assert results == ["scientific retrieval"]