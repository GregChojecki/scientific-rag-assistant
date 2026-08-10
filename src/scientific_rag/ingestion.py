from pathlib import Path


def list_pdf_files(folder: str | Path) -> list[Path]:
    """
    Return all PDF files found directly inside a folder.
    """
    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    return sorted(folder.glob("*.pdf"))