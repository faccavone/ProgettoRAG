from tqdm import tqdm
import os
from data_processing.pdf_loader import read_pdfs_from_folder
from data_processing.text_chunker import chunk_text
from data_processing.contextualizer import process_chunks_with_context
from data_processing.image_captioning import generate_image_caption

def prepare_contextualized_chunks(pdf_folder, chunk_size, overlap, window_size=10):
    """
    Estrae testo e immagini da tutti i PDF nella cartella.
    Genera:
    - chunk contestualizzati
    - caption per immagini
    Mostra progresso durante il processo.
    """
    pdfs = read_pdfs_from_folder(pdf_folder)
    results = []

    for pdf_name, full_text, images in tqdm(pdfs, desc="📄 Elaborazione PDF"):
        # 1️⃣ Chunking
        chunks = chunk_text(full_text, chunk_size, overlap)
        print(f"📑 {len(chunks)} chunk trovati in {pdf_name}")

        # 2️⃣ Contextualizzazione → barra è ora in contextualizer.py
        processed_chunks = process_chunks_with_context(chunks, window_size)

        for i, (chunk, context) in enumerate(processed_chunks):
            contextualized_chunk = f"{context}\n\n{chunk}"
            results.append((contextualized_chunk, pdf_name, "text", i, {}))

        # 3️⃣ Caption delle immagini
        for j, img in tqdm(enumerate(images), total=len(images), desc=f"🖼️ Captioning: {pdf_name}"):
            caption = generate_image_caption(img["path"])
            if caption:
                results.append((
                    caption, pdf_name, "image", j, {
                        "image_path": os.path.abspath(img["path"]),
                        "page": img["page"]
                    }
                ))

    return results
