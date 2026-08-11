import chromadb


def create_chroma_client() -> chromadb.ClientAPI:
    """
    Create and return an in-memory ChromaDB client.
    """
    return chromadb.Client()


def add_documents(
    client: chromadb.ClientAPI,
    collection_name: str,
    documents: list[str],
    embeddings,
) -> chromadb.Collection:
    """
    Store documents and their embeddings in a ChromaDB collection.
    """
    collection = client.get_or_create_collection(name=collection_name)

    ids = [f"doc_{i}" for i in range(len(documents))]

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings,
    )

    return collection