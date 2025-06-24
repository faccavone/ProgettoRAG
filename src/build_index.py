from retrieval.add_to_chroma import add_documents_to_chroma
from retrieval.build_bm25_index import build_bm25_index
from data_processing.prepare_chunks import prepare_contextualized_chunks
from config.settings import CHUNK_SIZE, OVERLAP, PDF_FOLDER
import os

import warnings
from urllib3.exceptions import NotOpenSSLWarning

warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

if __name__ == "__main__":
    processed_results_with_sources = prepare_contextualized_chunks(
       PDF_FOLDER, CHUNK_SIZE, OVERLAP, window_size=10
   )

    # Inizializza Pinecone e BM25 con i chunk processati
    add_documents_to_chroma(processed_results_with_sources)
    build_bm25_index(processed_results_with_sources)
