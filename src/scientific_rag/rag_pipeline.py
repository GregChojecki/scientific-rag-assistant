from pathlib import Path

from .chunking import chunk_text
from .embeddings import embed_texts, load_embedding_model
from .generation import generate_answer
from .ingestion import extract_text_from_pdf
from .retrieval import retrieve_documents
from .vector_store import add_documents, create_chroma_client


def build_retrieval_collection(pdf_path: str | Path):
    """
    Build an in-memory ChromaDB collection from a PDF.
    """
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)

    metadatas = [
        {
            "source": Path(pdf_path).name,
            "chunk_id": i,
        }
        for i in range(len(chunks))
    ]

    model = load_embedding_model()
    embeddings = embed_texts(chunks, model)

    client = create_chroma_client()

    collection = add_documents(
        client=client,
        collection_name="scientific_documents",
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return collection, model


def retrieve_context(
    question: str,
    collection,
    model,
    n_results: int = 3,
):
    """
    Embed a question and retrieve relevant chunks with metadata.
    """
    query_embedding = embed_texts([question], model)[0]

    results = retrieve_documents(
        collection=collection,
        query_embedding=query_embedding,
        n_results=n_results,
    )

    return [
        {
            "text": document,
            "metadata": metadata,
        }
        for document, metadata in results
    ]


def answer_question(
    question: str,
    collection,
    model,
    n_results: int = 3,
) -> str:
    """
    Retrieve relevant context and generate a grounded answer.
    """
    retrieved_context = retrieve_context(
        question=question,
        collection=collection,
        model=model,
        n_results=n_results,
    )

    return generate_answer(
        question=question,
        context_chunks=[item["text"] for item in retrieved_context],
    )