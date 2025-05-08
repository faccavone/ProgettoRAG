import os
from groq import Groq
from config.settings import GROQ_API_KEY

# Inizializza il client GROQ
client = Groq(api_key=GROQ_API_KEY)

def generate_answer(context: str, question: str) -> str:
    """
    Usa GROQ e LLaMA 4 Scout per generare una risposta basata sul contesto fornito.
    """
    # Prompt con istruzioni dettagliate
    prompt = f"""
    Sei un assistente AI specializzato in Recupero Aumentato di Informazioni (RAG). 
    La tua funzione è fornire risposte **solo** in base al contesto fornito, senza inventare o aggiungere contenuti esterni.

    📚 Contesto:
    {context}

    ❓ Domanda:
    {question}

    ✍️ Istruzioni:
    - Analizza attentamente il contesto.
    - Rispondi con chiarezza, precisione e struttura logica.
    - Evita ripetizioni e ridondanze.
    - Se il contesto non è sufficiente per rispondere, dichiara esplicitamente che non ci sono abbastanza informazioni.

    📌 **Risposta:**  
        """

    messages = [
        {"role": "system", "content": "Sei un assistente specializzato in domande tecniche con capacità avanzate di comprensione del contesto."},
        {"role": "user", "content": prompt.strip()}
    ]

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            temperature=0.3,
            max_tokens=700
        )
        answer = response.choices[0].message.content.strip()

        # Pulizia opzionale
        marker = "📌 **Risposta:**"
        if marker in answer:
            return answer.split(marker)[1].strip()
        else:
            return answer

    except Exception as e:
        return f"⚠️ Errore durante la generazione con GROQ: {e}"
