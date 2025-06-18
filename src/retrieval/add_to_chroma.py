import uuid
from datetime import datetime
from embeddings.model import generate_embedding
from database.chroma_client import collection  # Assicurati di avere questo
from tqdm import tqdm

def add_documents_to_chroma(processed_results_with_sources, batch_size=100):
    """
    Riceve una lista di tuple:
        (contenuto, source, tipo, indice, metadati opzionali)
    e li salva su ChromaDB.
    """
    documents = []
    embeddings = []
    metadatas = []
    ids = []

    for item in tqdm(processed_results_with_sources, desc="📦 Indicizzazione in corso"):
        content, source, doc_type, index, metadata = item
        embedding = generate_embedding(content)

        if embedding:
            doc_id = str(uuid.uuid4())
            documents.append(content)
            embeddings.append(embedding)
            metadatas.append({
                "source": source,
                "type": doc_type,
                "index": index,
                "created_at": datetime.now().isoformat(),
                **metadata  # include image_path, page, ecc.
            })
            ids.append(doc_id)

    # Salvataggio in batch nel vector store
    for i in range(0, len(documents), batch_size):
        collection.add(
            documents=documents[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )
        print(f"✅ Aggiunto batch di {len(documents[i:i+batch_size])} elementi a ChromaDB.")
