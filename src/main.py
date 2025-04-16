from ui.interface import start_interface
from database.pdf_to_pinecone import add_pdfs_to_pinecone
from retrieval.build_bm25_index import build_bm25_index

if __name__ == "__main__":
    # ⚠️ Esegui questi due una sola volta per inizializzare Pinecone e BM25
    #add_pdfs_to_pinecone()
    #build_bm25_index()

    # Avvia l’interfaccia Gradio
    start_interface()
