import subprocess
from tqdm import tqdm

def build_chunk_context_window(chunks: list[str], current_index: int, window_size: int = 10) -> str:
    """
    Crea una ‘finestra’ di chunk attorno a quello corrente, estesa ± window_size.
    """

    start = max(0, current_index - window_size)
    end = min(len(chunks), current_index + window_size + 1)
    return "\n".join(chunks[start:end])

def generate_chunk_context(context_window: str, target_chunk: str) -> str:
    """
    Usa Ollama con llama3 per generare un contesto breve e comprensibile
    del chunk all’interno del documento.
    """

    prompt = f"""
    <document>
    {context_window}
    </document>

    Qui c'è il chunk che vogliamo contestualizzare all'interno dell'intero documento:
    <chunk>
    {target_chunk}
    </chunk>

    Per favore, fornisci un contesto breve e chiaro che descriva la posizione e il significato di questo chunk
    nel contesto complessivo del documento, in modo da migliorarne il recupero durante la ricerca. 
    Rispondi esclusivamente con il contesto richiesto, senza aggiungere altro.
    """

    try:
        # chiama il modello locale via Ollama
        result = subprocess.run(
            ["ollama", "run", "llama3"],
            input=prompt.encode("utf-8"),
            capture_output=True,
            check=True
        )
        return result.stdout.decode("utf-8").strip()

    except subprocess.CalledProcessError as e:
        # se c’è un errore, stampalo e torna una stringa di errore leggibile
        print(f"❌ Errore nella generazione del contesto: {e.stderr.decode()}")
        return f"⚠️ Errore: {e}"

def process_chunks_with_context(chunks: list[str], window_size: int = 10):
    """
    Genera contesto per ogni chunk, mostrando progresso con tqdm.
    """
    results = []
    for index, chunk in tqdm(enumerate(chunks), total=len(chunks), desc="🧠 Generazione contesto"):
        context_window = build_chunk_context_window(chunks, index, window_size)
        context = generate_chunk_context(context_window, chunk)
        results.append((chunk, context))
    return results