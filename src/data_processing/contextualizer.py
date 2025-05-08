import time
from groq import Groq
from config.settings import GROQ_API_KEY

# 🌐 Inizializzazione del client GROQ
client = Groq(api_key=GROQ_API_KEY)

def build_chunk_context_window(chunks: list[str], current_index: int, window_size: int = 3) -> str:
    """
    Costruisce un contesto limitato ai chunk vicini al chunk corrente.
    Include i `window_size` chunk precedenti e successivi rispetto al chunk in posizione `current_index`.
    """
    start = max(0, current_index - window_size)
    end = min(len(chunks), current_index + window_size + 1)
    context_chunks = chunks[start:end]
    
    return "\n".join(context_chunks)

def generate_chunk_context(context_window: str, target_chunk: str) -> str:
    # 📌 Prompt con marker
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

📌 **Risposta:**
    """

    # Creazione del payload da inviare
    messages = [
        {
            "role": "system",
            "content": "Sei un assistente che analizza documenti e genera contesto per dei frammenti di testo. Rispondi sempre con una descrizione concisa."
        },
        {
            "role": "user",
            "content": prompt.strip()
        }
    ]

    try:
        # Invio della richiesta al modello utilizzando GROQ
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="gemma2-9b-it",  # Utilizza il modello che desideri
            max_completion_tokens=100,  # Imposta i token massimi
            temperature=0.3
        )

        # Estrai il risultato dalla risposta
        raw_text = chat_completion.choices[0].message.content.strip()

        # 🧼 Pulizia del testo generato
        marker = "📌 **Risposta:**"
        if marker in raw_text:
            cleaned_text = raw_text.split(marker)[1].strip()
        else:
            cleaned_text = raw_text

        return cleaned_text

    except Exception as e:
        print(f"❌ Errore nella generazione del contesto: {e}")
        return f"⚠️ Errore: {e}"

def process_chunks_with_pause(chunks: list[str], window_size: int = 3, chunk_limit_per_minute: int = 3):
    """
    Elabora i chunk in modo che non vengano superati i limiti di token per minuto.
    Dopo ogni 4 chiamate, attende 1 minuto.
    """
    call_count = 0  # Conta le chiamate effettuate
    results = []

    for index, chunk in enumerate(chunks):
        # Creazione della finestra di contesto per il chunk corrente
        context_window = build_chunk_context_window(chunks, index, window_size)

        # Generazione del contesto per il chunk
        context = generate_chunk_context(context_window, chunk)
        results.append((chunk, context))

        call_count += 1

        # Se sono state fatte 4 chiamate, aspetta 1 minuto prima di continuare
        if call_count >= chunk_limit_per_minute:
            print("🕒 Limite di chiamate al minuto raggiunto. Pausa di 1 minuto...")
            time.sleep(60)  # Pausa di 1 minuto
            call_count = 0  # Reset del contatore delle chiamate

    return results
