import subprocess

def generate_image_caption(image_path: str) -> str:
    """
    Usa Ollama (modello LLaVA) per generare una descrizione dettagliata dell’immagine.

    Args:
        image_path (str): Percorso al file immagine

    Returns:
        str: Caption descrittiva generata dal modello
    """
    
    # Prompt per il modello LLaVA
    # Ottimizzato per generare descrizioni utili per la ricerca e il recupero
    prompt = """
    Sei un assistente incaricato di riassumere immagini per scopi di ricerca e recupero.
    Questi riassunti saranno utilizzati per generare embedding e servire a ritrovare l’immagine originale.
    Fornisci un riassunto conciso dell’immagine, ottimizzato per la ricerca.
    """
    try:
        result = subprocess.run(
            ["ollama", "run", "llava", "--image", image_path],
            input=prompt.encode("utf-8"),
            capture_output=True,
            check=True
        )
        return result.stdout.decode("utf-8").strip() 

    except subprocess.CalledProcessError as e:
        print(f"❌ Errore nella generazione della descrizione immagine: {e.stderr.decode(errors='ignore')}")
        return f"⚠️ Errore nella descrizione immagine"
