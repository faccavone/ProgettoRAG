import uuid, os, datetime
from database.pinecone_client import index
from embeddings.model import generate_embedding

def add_pdfs_to_pinecone(processed_results_with_sources, batch_size=100):
    """
    Riceve una lista di tuple (contextualized_chunk, source, chunk_index) e li salva su Pinecone.
    """
    upserts = []

    for contextualized_chunk, pdf_name, i in processed_results_with_sources:
        embedding = generate_embedding(contextualized_chunk)
        if embedding:
            upserts.append((str(uuid.uuid4()), embedding, {
                "text": contextualized_chunk,
                "source": pdf_name,
                "chunk_index": i,
                "created_at": datetime.datetime.now().isoformat()
            }))

    for i in range(0, len(upserts), batch_size):
        batch = upserts[i:i + batch_size]
        index.upsert(vectors=batch, namespace="default")
        print(f"✅ Aggiunti {len(batch)} chunk contestualizzati a Pinecone.")
