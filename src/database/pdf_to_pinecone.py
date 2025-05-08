import uuid, os, datetime
from concurrent.futures import ThreadPoolExecutor
from database.pinecone_client import index
from embeddings.model import generate_embedding
from data_processing.pdf_loader import read_pdfs_from_folder
from data_processing.text_chunker import chunk_text
from data_processing.contextualizer import generate_chunk_context, build_chunk_context_window, process_chunks_with_pause
from config.settings import CHUNK_SIZE, OVERLAP, PDF_FOLDER

def add_pdfs_to_pinecone(folder_path=PDF_FOLDER, chunk_size=CHUNK_SIZE, overlap=OVERLAP, batch_size=100, chunk_limit_per_minute=3):
    """
    Legge i PDF dalla cartella, li suddivide in chunk e li salva su Pinecone.
    """

    pdfs = read_pdfs_from_folder(folder_path)
    upserts = []

    def process_pdf(pdf_name, text):
        chunks = chunk_text(text, chunk_size, overlap)

        # Usa la funzione con la pausa
        processed_results = process_chunks_with_pause(chunks, window_size=3, chunk_limit_per_minute=chunk_limit_per_minute)

        for chunk, context in processed_results:
            contextualized_chunk = f"{context}\n\n{chunk}"
            embedding = generate_embedding(contextualized_chunk)
            if embedding:
                upserts.append((str(uuid.uuid4()), embedding, {
                    "text": contextualized_chunk,
                    "source": pdf_name,
                    "chunk_index": i,
                    "created_at": datetime.now().isoformat()
                }))

    with ThreadPoolExecutor() as executor:
        executor.map(lambda pdf: process_pdf(*pdf), pdfs)

    for i in range(0, len(upserts), batch_size):
        batch = upserts[i:i + batch_size]
        index.upsert(vectors=batch, namespace="default")
        print(f"✅ Aggiunti {len(batch)} chunk contestualizzati a Pinecone.")
