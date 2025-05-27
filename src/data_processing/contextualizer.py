import time
import google.generativeai as genai
from config.settings import GEMINI_API_KEY

# 🌐 Configura l'autenticazione
genai.configure(api_key=GEMINI_API_KEY)

# Inizializza il modello Gemini 2.5 Flash
model = genai.GenerativeModel("gemini-2.0-flash")

def build_chunk_context_window(chunks: list[str], current_index: int, window_size: int = 10) -> str:
    start = max(0, current_index - window_size)
    end = min(len(chunks), current_index + window_size + 1)
    return "\n".join(chunks[start:end])

def generate_chunk_context(context_window: str, target_chunk: str) -> str:
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
        response = model.generate_content(
            prompt.strip(),
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 100,
            }
        )

        return response.text.strip()

    except Exception as e:
        print(f"❌ Errore nella generazione del contesto: {e}")
        return f"⚠️ Errore: {e}"

def process_chunks_with_pause(chunks: list[str], window_size: int = 10, chunk_limit_per_minute: int = 10):
    call_count = 0
    results = []

    for index, chunk in enumerate(chunks):
        context_window = build_chunk_context_window(chunks, index, window_size)
        context = generate_chunk_context(context_window, chunk)
        results.append((chunk, context))

        call_count += 1
        if call_count >= chunk_limit_per_minute:
            print("🕒 Limite di chiamate al minuto raggiunto. Pausa di 1 minuto...")
            time.sleep(60)
            call_count = 0

    return results
