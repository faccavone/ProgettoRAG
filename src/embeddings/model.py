from sentence_transformers import SentenceTransformer

# Carica il modello di embedding all-mpnet-base-v2
model = SentenceTransformer("all-mpnet-base-v2")

def generate_embedding(text):
    """
    Genera l'embedding di un dato testo.

    Args:
        text (str): Testo da convertire in embedding.

    Returns:
        list: Lista di valori float rappresentante il vettore embedding.
    """
    return model.encode(text).tolist()
