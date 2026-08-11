from pathlib import Path

from .chunking import chunk_text
from .embeddings import embed_texts, load_embedding_model
from .ingestion import extract_text_from_pdf
from .vector_store import add_documents, create_chroma_client


def build_retrieval_collection(pdf_path: str | Path):
    """
    Build an in-memory ChromaDB collection from a PDF.
    """
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)

    model = load_embedding_model()
    embeddings = embed_texts(chunks, model)

    client = create_chroma_client()

    collection = add_documents(
        client=client,
        collection_name="scientific_documents",
        documents=chunks,
        embeddings=embeddings,
    )

    return collection, model