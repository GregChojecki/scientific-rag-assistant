def retrieve_documents(
    collection,
    query_embedding,
    n_results: int = 3,
):
    """
    Retrieve the most similar documents from a ChromaDB collection.
    """
    result = collection.query(
        query_embeddings=[
            query_embedding.tolist()
            if hasattr(query_embedding, "tolist")
            else query_embedding
        ],
        n_results=n_results,
    )

    return result["documents"][0]