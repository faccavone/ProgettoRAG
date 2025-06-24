import requests
import base64
import json

def generate_image_caption(image_path: str) -> str:
    """
    Usa LLaVA via Ollama API per generare descrizione dell'immagine (gestione risposta NDJSON).
    """
    prompt = (
        "Sei un assistente incaricato di riassumere immagini per scopi di ricerca e recupero. "
        "Questi riassunti saranno utilizzati per generare embedding e servire a ritrovare l’immagine originale. "
        "Fornisci un riassunto conciso dell’immagine, ottimizzato per la ricerca."
    )

    try:
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")

        payload = {
            "model": "llava",
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_data]
                }
            ]
        }

        response = requests.post("http://localhost:11434/api/chat", json=payload)
        response.raise_for_status()

        # La risposta è in NDJSON: parse riga per riga
        full_content = ""
        done = False

        for line in response.text.strip().splitlines():
            data = json.loads(line)
            content = data.get("message", {}).get("content", "")
            full_content += content
            if data.get("done", False):
                done = True
                break

        if not done:
            print("⚠️ Attenzione: risposta incompleta (done: false)")

        return full_content.strip()

    except Exception as e:
        print(f"❌ Errore nella generazione della descrizione immagine: {e}")
        return "⚠️ Errore nella descrizione immagine"
