import os
import PyPDF2

def read_pdfs_from_folder(folder_path):
    """
    Legge tutti i file PDF in una cartella ed estrae il testo.

    Args:
        folder_path (str): Percorso della cartella con i PDF.

    Returns:
        list: Lista di tuple (nome_file, testo_estratto)
    """
    pdf_texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
                pdf_texts.append((filename, text))
    return pdf_texts
