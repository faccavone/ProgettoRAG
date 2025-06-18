import os
import pickle
import spacy
from rank_bm25 import BM25Okapi
from config.settings import BM25_INDEX_PATH

nlp = spacy.load("it_core_news_sm")  # Cambia se usi l'inglese

def build_bm25_index(processed_results_with_sources):
    """
    Costruisce l'indice BM25 da testo e descrizioni immagine.
    Accetta lista di tuple: (contenuto, source, type, index, metadati)
    """
    corpus = []
    sources = []

    for content, name, doc_type, i, metadata in processed_results_with_sources:
        # Usa solo contenuto testuale (sia chunk che descrizioni immagine)
        corpus.append(content)
        sources.append({
            "text": content,
            "source": name,
            "type": doc_type,
            "index": i,
            **metadata  # include ad esempio image_path, page, ecc.
        })

    # Tokenizza il testo per BM25
    tokenized_corpus = [
        [token.text for token in nlp(doc) if not token.is_space]
        for doc in corpus
    ]
    bm25 = BM25Okapi(tokenized_corpus)

    # Salva indice e dati associati
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "documents": sources}, f)

    print(f"✅ BM25 index costruito con {len(corpus)} elementi (inclusi testo + immagini).")
