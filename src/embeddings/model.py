from sentence_transformers import SentenceTransformer

# Carica il modello di embedding
model = SentenceTransformer("all-MiniLM-L6-v2")

def generate_embedding(text):
    """
    Genera l'embedding di un dato testo.

    Args:
        text (str): Testo da convertire in embedding.

    Returns:
        list: Lista di valori float rappresentante il vettore embedding.
    """
    return model.encode(text).tolist()
