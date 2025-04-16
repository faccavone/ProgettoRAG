from sentence_transformers import CrossEncoder
from config.settings import TOP_K
# Scegli un modello bilanciato tra accuratezza e performance
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-4-v2"
reranker = CrossEncoder(MODEL_NAME)

def rerank_documents(query, documents, top_k=TOP_K):
    """
    Rerank dei chunk usando un cross-encoder.

    Args:
        query (str): Domanda dell’utente.
        documents (list of dict): Ogni dict ha "text" e "source".
        top_k (int): Numero finale di chunk da mantenere.

    Returns:
        list of tuple: (text, source) dei documenti rerankati.
    """

    # Prepara coppie (query, testo) per il reranker
    pairs = [(query, doc["text"]) for doc in documents]

    # Calcola punteggi di rilevanza
    scores = reranker.predict(pairs, batch_size=24)  

    # Aggiungi punteggio ai documenti
    for i, score in enumerate(scores):
        documents[i]["rerank_score"] = float(score)

    # Ordina per punteggio decrescente
    ranked = sorted(documents, key=lambda d: d["rerank_score"], reverse=True)

    return [(doc["text"], doc["source"], doc["rerank_score"]) for doc in ranked[:top_k]]
