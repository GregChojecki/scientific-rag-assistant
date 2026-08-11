from src.scientific_rag.chunking import chunk_text


def test_chunk_text_creates_overlapping_chunks():
    text = "abcdefghij"

    chunks = chunk_text(text, chunk_size=5, overlap=2)

    assert chunks == ["abcde", "defgh", "ghij"]