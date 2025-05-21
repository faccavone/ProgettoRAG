import os
import pickle
import spacy
from rank_bm25 import BM25Okapi
from config.settings import CHUNK_SIZE, OVERLAP, BM25_INDEX_PATH, PDF_FOLDER


nlp = spacy.load("it_core_news_sm")  # Cambia in "en_core_web_sm" se lavori in inglese it_core_news_sm

def build_bm25_index(processed_results_with_sources):
    """
    Costruisce l'indice BM25 a partire da una lista di tuple (contextualized_chunk, source, chunk_index).
    """
    corpus = []
    sources = []

    for contextualized_chunk, name, i in processed_results_with_sources:
        corpus.append(contextualized_chunk)
        sources.append({
            "text": contextualized_chunk,
            "source": name,
            "chunk_index": i
        })

    # Tokenizza i chunk (BM25 lavora con liste di parole)
    tokenized_corpus = [[token.text for token in nlp(doc) if not token.is_space] for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    # Salva l'indice BM25 e i chunk associati su disco
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "documents": sources}, f)

    print(f"✅ BM25 index costruito con {len(corpus)} chunk contestualizzati.")

