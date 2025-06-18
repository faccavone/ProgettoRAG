import pickle
from embeddings.model import generate_embedding
from config.settings import TOP_K, SIMILARITY_THRESHOLD, BM25_INDEX_PATH, INITIAL_K
from rank_bm25 import BM25Okapi
from retrieval.reranker import rerank_documents
from database.chroma_client import collection  # Vector store locale (Chroma)

# === 🔁 BM25 SEARCH ===

def load_bm25_index():
    """
    Carica l’indice BM25 e i documenti indicizzati.
    """
    with open(BM25_INDEX_PATH, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["documents"]


def bm25_search(query, top_k):
    """
    Cerca i top-k contenuti usando BM25, inclusi chunk testuali e descrizioni immagine.
    """
    bm25, documents = load_bm25_index()
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

    return [
        {
            "text": documents[i]["text"],
            "source": documents[i]["source"],
            "type": documents[i].get("type", "text"),
            "score": s,
            "rank": rank + 1,
            "image_path": documents[i].get("image_path", None)
        }
        for rank, (i, s) in enumerate(ranked)
    ]

# === 🧠 EMBEDDING SEARCH ===

def embedding_search(query, top_k):
    """
    Cerca i top-k contenuti semanticamente simili usando ChromaDB.
    Include anche descrizioni di immagini.
    """
    embedding = generate_embedding(query)
    if not embedding:
        return []

    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["metadatas", "documents"]
    )

    matches = []
    for i, doc in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][i]
        matches.append({
            "text": doc,
            "source": metadata.get("source", ""),
            "type": metadata.get("type", "text"),
            "score": results["distances"][0][i],  # distanza (può essere trasformata)
            "rank": i + 1,
            "image_path": metadata.get("image_path", None)
        })

    return matches

# === 🔁 FUSIONE ===

def reciprocal_rank_fusion(results_list, k=60):
    """
    Combina più liste (BM25, Embedding) con Reciprocal Rank Fusion (RRF).
    Mantiene metadati utili per il post-processing.
    """
    fused_scores = {}

    for results in results_list:
        for doc in results:
            doc_text = doc["text"]
            rank = doc["rank"]
            score_rrf = 1 / (k + rank)

            if doc_text in fused_scores:
                fused_scores[doc_text]["score"] += score_rrf
            else:
                fused_scores[doc_text] = {
                    "text": doc_text,
                    "source": doc["source"],
                    "type": doc.get("type", "text"),
                    "image_path": doc.get("image_path"),
                    "score": score_rrf
                }

    return sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)

# === 🔎 ENTRY POINT ===

def search_documents(query, top_k=TOP_K, initial_k=INITIAL_K):
    """
    Esegue una ricerca combinata BM25 + Embedding con reranking finale.
    Restituisce lista di dizionari con testo, fonte, tipo e immagine se presente.
    """
    if len(query) < 3:
        return [{"text": "⚠️ La query è troppo breve per una ricerca significativa."}]

    # 🔹 Recupero iniziale da entrambi i motori
    emb_results = embedding_search(query, initial_k)
    bm25_results = bm25_search(query, initial_k)

    # 🔹 Fusione con Reciprocal Rank Fusion
    fused_results = reciprocal_rank_fusion([emb_results, bm25_results])

    # 🔹 Reranking con CrossEncoder
    reranked_results = rerank_documents(query, fused_results, top_k=top_k)

    # 🔹 Ricostruisci output finale con metadati da fused_results
    final_results = []
    for (text, source, score) in reranked_results:
        match = next((d for d in fused_results if d["text"] == text and d["source"] == source), None)
        if match and score >= SIMILARITY_THRESHOLD:
            final_results.append({
                "text": text,
                "source": source,
                "score": score,
                "type": match.get("type", "text"),
                "image_path": match.get("image_path")
            })

    return final_results[:top_k]
