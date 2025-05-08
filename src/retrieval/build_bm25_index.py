import os
import pickle
import spacy
from rank_bm25 import BM25Okapi
from data_processing.pdf_loader import read_pdfs_from_folder
from data_processing.text_chunker import chunk_text
from data_processing.contextualizer import generate_chunk_context, build_chunk_context_window, process_chunks_with_pause
from config.settings import CHUNK_SIZE, OVERLAP, BM25_INDEX_PATH, PDF_FOLDER


nlp = spacy.load("it_core_news_sm")  # Cambia in "en_core_web_sm" se lavori in inglese it_core_news_sm

def build_bm25_index(chunk_limit_per_minute=3):
    pdfs = read_pdfs_from_folder(PDF_FOLDER)
    corpus = []  # Lista di chunk (con contesto)
    sources = []  # Metadati: testo + nome file sorgente

    for name, text in pdfs:
        chunks = chunk_text(text, CHUNK_SIZE, OVERLAP)

        # Usa la funzione con la pausa
        processed_results = process_chunks_with_pause(chunks, window_size=3, chunk_limit_per_minute=chunk_limit_per_minute)

        for i, (chunk, context) in enumerate(processed_results):
            contextualized_chunk = f"{context}\n\n{chunk}"
            corpus.append(contextualized_chunk)
            sources.append({
                "text": contextualized_chunk,
                "source": name,
                "chunk_index": i  # Usa l'indice di enumerate
            })

    # Tokenizza i chunk (BM25 lavora con liste di parole)
    tokenized_corpus = [[token.text for token in nlp(doc) if not token.is_space] for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    # Salva l'indice BM25 e i chunk associati su disco
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "documents": sources}, f)

    print(f"✅ BM25 index costruito con {len(corpus)} chunk contestualizzati.")

