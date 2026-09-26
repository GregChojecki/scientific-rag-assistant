from unittest.mock import patch

from src.scientific_rag.generation import generate_answer


@patch("src.scientific_rag.generation.subprocess.run")
def test_generate_answer_uses_context(mock_run):
    mock_run.return_value.stdout = "Moisture content was predicted using HSI and PLSR."

    answer = generate_answer(
        question="How was moisture content predicted?",
        context_chunks=[
            "Hyperspectral imaging and partial least squares regression were used."
        ],
    )

    assert "HSI" in answer
    assert "PLSR" in answer
    mock_run.assert_called_once()