from src.scientific_rag.ingestion import list_pdf_files


def test_list_pdf_files_returns_empty_list_for_empty_folder(tmp_path):
    result = list_pdf_files(tmp_path)

    assert result == []