def chunk_text(text, chunk_size, overlap):
    """
    Divide il testo in chunk sovrapposti.

    Args:
        text (str): Testo da dividere.
        chunk_size (int): Numero di parole per chunk.
        overlap (int): Numero di parole che si sovrappongono tra chunk.

    Returns:
        list: Lista di chunk di testo.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunk = " ".join(words[start:start + chunk_size])
        chunks.append(chunk)
        start += chunk_size - overlap  
    return chunks
