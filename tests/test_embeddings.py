from src.scientific_rag.embeddings import load_embedding_model, embed_texts


def test_embed_texts_returns_expected_shape():
    model = load_embedding_model()

    embeddings = embed_texts(
        ["machine learning", "scientific document retrieval"],
        model,
    )

    assert embeddings.shape == (2, 384)