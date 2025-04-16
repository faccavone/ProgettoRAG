import pickle
from embeddings.model import generate_embedding
from database.pinecone_client import index
from config.settings import TOP_K, SIMILARITY_THRESHOLD, BM25_INDEX_PATH, INITIAL_K
from rank_bm25 import BM25Okapi
from retrieval.reranker import rerank_documents 


def load_bm25_index():
    """
    Carica l’indice BM25 e i chunk salvati.
    """
    with open(BM25_INDEX_PATH, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["documents"]


def bm25_search(query, top_k):
    """
    Cerca i top-k chunk usando BM25.
    """
    bm25, documents = load_bm25_index()
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    
    return [
        {"text": documents[i]["text"], "source": documents[i]["source"], "score": s, "rank": rank+1}
        for rank, (i, s) in enumerate(ranked)
    ]


def embedding_search(query, top_k):
    """
    Cerca i top-k chunk semanticamente simili usando Pinecone.
    """
    embedding = generate_embedding(query)
    if not embedding:
        return []

    results = index.query(
        namespace="default",
        vector=embedding,
        top_k=top_k,
        include_metadata=True
    )

    return [
        {
            "text": match["metadata"]["text"],
            "source": match["metadata"]["source"],
            "score": match["score"],
            "rank": rank + 1  # Rank basato sulla posizione nel risultato
        }
        for rank, match in enumerate(results.get("matches", []))
    ]


def reciprocal_rank_fusion(results_list, k=60):
    """
    Applica Reciprocal Rank Fusion (RRF) per combinare le classifiche.
    
    Args:
        results_list (list of lists): Liste di documenti provenienti da diverse fonti (BM25, Embedding).
        k (int): Costante per RRF (default=60).

    Returns:
        list: Documenti ordinati per punteggio RRF.
    """
    fused_scores = {}

    for results in results_list:
        for doc in results:
            doc_text = doc["text"]
            rank = doc["rank"]

            # Calcolo RRF Score
            score_rrf = 1 / (k + rank)

            # Somma i punteggi RRF
            if doc_text in fused_scores:
                fused_scores[doc_text]["score"] += score_rrf
            else:
                fused_scores[doc_text] = {"text": doc_text, "source": doc["source"], "score": score_rrf}

    # Ordina i documenti per punteggio RRF decrescente
    sorted_results = sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)

    return sorted_results


def search_documents(query, top_k=TOP_K, initial_k = INITIAL_K):
    """
    Recupera i chunk più rilevanti combinando BM25 e embedding con Reciprocal Rank Fusion.

    Args:
        query (str): Domanda in input.
        top_k (int): Numero massimo di risultati.

    Returns:
        list of tuple: (testo, fonte) dei documenti rilevanti.
    """
    if len(query) < 3:
        return ["⚠️ La query è troppo breve per una ricerca significativa."]

    print(f"\n🔍 Eseguo ricerca per: {query}")

    # 🔹 Recupero documenti da entrambi i motori di ricerca, recupero iniziale
    emb_results = embedding_search(query, initial_k)
    bm25_results = bm25_search(query, initial_k)

    # 🔹 Fusione dei risultati con Rank Fusion
    fused_results = reciprocal_rank_fusion([emb_results, bm25_results])

    # Reranking finale
    reranked_results = rerank_documents(query, fused_results, top_k=top_k)

    # Filtro sui risultati rerankati
    filtered_results = [
        (text, source, score)
        for (text, source, score) in reranked_results
        if score >= SIMILARITY_THRESHOLD
    ]

    print("\n📚 Chunk Selezionati (Dopo Reranking + Filtro):")
    for i, (text, source, score) in enumerate(filtered_results, 1):
        print(f"{i}. Source: {source} - Testo: {text[:1000]} - Score {score}\n")

    if not filtered_results:
        return ["❌ Nessun documento supera la soglia di similarità impostata."]

    return filtered_results[:top_k]
