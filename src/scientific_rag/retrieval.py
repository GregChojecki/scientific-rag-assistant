def retrieve_documents(
    collection,
    query_embedding,
    n_results: int = 3,
):
    """
    Retrieve the most similar documents and metadata from a ChromaDB collection.
    """
    result = collection.query(
        query_embeddings=[
            query_embedding.tolist()
            if hasattr(query_embedding, "tolist")
            else query_embedding
        ],
        n_results=n_results,
        include=["documents", "metadatas"],
    )

    return list(
        zip(
            result["documents"][0],
            result["metadatas"][0],
        )
    )