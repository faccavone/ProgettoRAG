import uuid, os, datetime
from concurrent.futures import ThreadPoolExecutor
from database.pinecone_client import index
from embeddings.model import generate_embedding
from data_processing.pdf_loader import read_pdfs_from_folder
from data_processing.text_chunker import chunk_text
from data_processing.contextualizer import generate_chunk_context
from config.settings import CHUNK_SIZE, OVERLAP, PDF_FOLDER

def add_pdfs_to_pinecone(folder_path=PDF_FOLDER, chunk_size=CHUNK_SIZE, overlap=OVERLAP, batch_size=100):
    """
    Legge i PDF dalla cartella, li suddivide in chunk e li salva su Pinecone.

    Args:
        folder_path (str): Percorso della cartella contenente i PDF.
        chunk_size (int): Dimensione di ogni chunk di testo.
        overlap (int): Sovrapposizione tra i chunk per il contesto.
        batch_size (int): Numero di chunk da inviare a Pinecone per volta.
    """
    pdfs = read_pdfs_from_folder(folder_path)
    upserts = []

    def process_pdf(pdf_name, text):
        chunks = chunk_text(text, chunk_size, overlap)
        for i, chunk in enumerate(chunks):
            context = generate_chunk_context(text, chunk)
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
        batch = upserts[i:i+batch_size]
        index.upsert(vectors=batch, namespace="default")
        print(f"✅ Aggiunti {len(batch)} chunk contestualizzati a Pinecone.")