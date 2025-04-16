import requests
from config.settings import HUGGINGFACE_API_KEY

API_URL = "http://localhost:1234/v1/chat/completions"

def generate_chunk_context(document: str, chunk: str) -> str:
    # Aggiungi il marker "📌 **Risposta:**" nel prompt
    prompt = f"""
<document>
{document}
</document>

Qui c'è il chunk che vogliamo contestualizzare all'interno dell'intero documento:
<chunk>
{chunk}
</chunk>

Per favore, fornisci un contesto breve e chiaro che descriva la posizione e il significato di questo chunk
nel contesto complessivo del documento, in modo da migliorarne il recupero durante la ricerca. 
Rispondi esclusivamente con il contesto richiesto, senza aggiungere altro.

📌 **Risposta:**
    """
    
    payload = {
    "model": "llama-3.2-1b-instruct",  # <-- assicurati di includere il nome del modello
    "messages": [
  {
    "role": "system",
    "content": "Sei un assistente che analizza documenti e genera contesto per dei frammenti di testo. Rispondi sempre con una descrizione concisa."
  },
  {
    "role": "user",
    "content": "<document>\n{document}\n</document>\n\n<chunk>\n{chunk}\n</chunk>\n\n📌 **Risposta:**"
  }
],
    "max_tokens": 100
    }

    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()

        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            raw_text = result["choices"][0]["message"]["content"].strip()
        else:
            return "⚠️ Errore: risposta vuota dal modello."

        # Estrai solo la parte dopo il marker (opzionale)
        marker = "📌 **Risposta:**"
        if marker in raw_text:
            cleaned_text = raw_text.split(marker)[1].strip()
        else:
            cleaned_text = raw_text

        return cleaned_text

    except requests.exceptions.RequestException as e:
        print(f"❌ Errore nella generazione del contesto: {e}")
        return f"⚠️ Errore nella richiesta API: {e}"
    except Exception as e:
        print(f"❌ Errore generico: {e}")
        return f"⚠️ Errore generico: {e}"