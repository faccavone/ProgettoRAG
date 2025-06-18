import os
from pathlib import Path
import fitz  # PyMuPDF

def read_pdfs_from_folder(folder_path, image_output_dir="extracted_images"):
    """
    Legge tutti i file PDF in una cartella, estrae il testo e le immagini per ogni pagina.

    Args:
        folder_path (str): Percorso della cartella con i PDF.
        image_output_dir (str): Cartella dove salvare le immagini estratte.

    Returns:
        list: Lista di tuple (nome_file, testo_estratto, immagini_estratte)
            - testo_estratto: stringa del testo unito
            - immagini_estratte: lista di dizionari per ogni immagine
    """
    os.makedirs(image_output_dir, exist_ok=True)
    pdf_data = []

    for filename in os.listdir(folder_path):
        if not filename.endswith(".pdf"):
            continue

        file_path = os.path.join(folder_path, filename)
        doc = fitz.open(file_path)
        full_text = []
        images = []

        for page_index, page in enumerate(doc):
            # ✅ Estrai il testo
            page_text = page.get_text()
            if page_text:
                full_text.append(page_text)

            # ✅ Estrai immagini dalla pagina
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_filename = f"{Path(filename).stem}_p{page_index}_img{img_index}.{image_ext}"
                image_path = os.path.join(image_output_dir, image_filename)

                with open(image_path, "wb") as f:
                    f.write(image_bytes)

                images.append({
                    "page": page_index,
                    "path": image_path,
                    "filename": image_filename,
                    "pdf_file": filename
                })

        pdf_data.append((filename, "\n".join(full_text), images))

    return pdf_data
