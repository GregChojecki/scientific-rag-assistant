from pathlib import Path

import pymupdf


def list_pdf_files(folder: str | Path) -> list[Path]:
    """
    Return all PDF files found directly inside a folder.
    """
    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    return sorted(folder.glob("*.pdf"))


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """
    Extract text from all pages of a PDF file.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")

    document = pymupdf.open(pdf_path)

    text = "\n".join(page.get_text() for page in document)

    document.close()

    return text