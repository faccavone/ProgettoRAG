from data_processing.pdf_loader import read_pdfs_from_folder
from data_processing.text_chunker import chunk_text
from data_processing.contextualizer import process_chunks_with_pause

def prepare_contextualized_chunks(pdf_folder, chunk_size, overlap, chunk_limit_per_minute=10, window_size=10):
    """
    Legge i PDF dalla cartella, li suddivide in chunk, aggiunge il contesto e restituisce una lista di tuple
    (contextualized_chunk, source, chunk_index).
    """
    pdfs = read_pdfs_from_folder(pdf_folder)
    processed_results_with_sources = []
    for name, text in pdfs:
        chunks = chunk_text(text, chunk_size, overlap)
        #Numero di chunk creati
        print(f"✅ {len(chunks)} chunk creati per il file {name}.")

        processed_results = process_chunks_with_pause(chunks, window_size=window_size, chunk_limit_per_minute=chunk_limit_per_minute)
        for i, (chunk, context) in enumerate(processed_results):
            contextualized_chunk = f"{context}\n\n{chunk}"
            processed_results_with_sources.append((contextualized_chunk, name, i))
    return processed_results_with_sources
