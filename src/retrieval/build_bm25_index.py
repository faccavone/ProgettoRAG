import os
import pickle
from rank_bm25 import BM25Okapi
from data_processing.pdf_loader import read_pdfs_from_folder
from data_processing.text_chunker import chunk_text
from data_processing.contextualizer import generate_chunk_context
from config.settings import CHUNK_SIZE, OVERLAP, BM25_INDEX_PATH, PDF_FOLDER


def build_bm25_index():
    # Legge tutti i PDF dalla cartella
    pdfs = read_pdfs_from_folder(PDF_FOLDER)
    corpus = []   # Lista di chunk (con contesto)
    sources = []  # Metadati: testo + nome file sorgente

    for name, text in pdfs:
        chunks = chunk_text(text, CHUNK_SIZE, OVERLAP)
        for i, chunk in enumerate(chunks):
            context = generate_chunk_context(text, chunk)
            contextualized_chunk = f"{context}\n\n{chunk}"
            corpus.append(contextualized_chunk)
            sources.append({
                "text": contextualized_chunk,
                "source": name,
                "chunk_index": i
            })

    # Tokenizza i chunk (BM25 lavora con liste di parole)
    tokenized_corpus = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    # Salva l'indice BM25 e i chunk associati su disco
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "documents": sources}, f)

    print(f"✅ BM25 index costruito con {len(corpus)} chunk contestualizzati.")

# Se eseguito come script standalone
if __name__ == "__main__":
    build_bm25_index()
