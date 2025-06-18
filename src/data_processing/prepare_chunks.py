from data_processing.pdf_loader import read_pdfs_from_folder
from data_processing.text_chunker import chunk_text
from data_processing.contextualizer import process_chunks_with_context
from data_processing.image_captioning import generate_image_caption

def prepare_contextualized_chunks(pdf_folder, chunk_size, overlap, window_size=10):
    """
    Estrae testo e immagini da tutti i PDF nella cartella.
    - Suddivide il testo in chunk con contesto (via LLaMA3)
    - Genera descrizioni delle immagini (via LLaVA)
    Restituisce lista di tuple:
        (contenuto, source_pdf, tipo, indice, metadati opzionali)
    """
    pdfs = read_pdfs_from_folder(pdf_folder)
    results = []

    for pdf_name, full_text, images in pdfs:
        # ✅ Gestione testo → chunk
        chunks = chunk_text(full_text, chunk_size, overlap)
        print(f"✅ {len(chunks)} chunk creati per il file {pdf_name}.")

    # Processa i chunk con contesto
        processed_chunks = process_chunks_with_context(chunks, window_size=window_size)
        for i, (chunk, context) in enumerate(processed_chunks):
            contextualized_chunk = f"{context}\n\n{chunk}"
            results.append((
                contextualized_chunk, pdf_name, "text", i, {}  # type 'text'
            ))

        # ✅ Gestione immagini → descrizione
        # per ogni immagine, genera una descrizione
        for j, img in enumerate(images):
            caption = generate_image_caption(img["path"])
            if caption:
                results.append((
                    caption, pdf_name, "image", j, {
                        "image_path": img["path"],
                        "page": img["page"]
                    }
                ))

    return results #
