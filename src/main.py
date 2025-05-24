from ui.interface import start_interface
from pdf_to_pinecone import add_pdfs_to_pinecone
from retrieval.build_bm25_index import build_bm25_index
from data_processing.prepare_chunks import prepare_contextualized_chunks
from config.settings import CHUNK_SIZE, OVERLAP, PDF_FOLDER

if __name__ == "__main__":
    #processed_results_with_sources = prepare_contextualized_chunks(
    #    PDF_FOLDER, CHUNK_SIZE, OVERLAP, chunk_limit_per_minute=10, window_size=10
    #)

    # Inizializza Pinecone e BM25 con i chunk processati
    # add_pdfs_to_pinecone(processed_results_with_sources)
    # build_bm25_index(processed_results_with_sources)

    # Avvia l’interfaccia Gradio
    start_interface()
