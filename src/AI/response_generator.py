import subprocess

def generate_answer(context: str, question: str) -> str:
    """
    Usa un modello LLaMA locale (via Ollama) per generare una risposta basata sul contesto fornito.
    Il modello risponde solo sulla base del contesto, in stile RAG.
    """

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

    try:
        result = subprocess.run(
            ["ollama", "run", "llama3"],  
            input=prompt.encode("utf-8"),
            capture_output=True,
            check=True
        )

        # Estrai la risposta
        raw_output = result.stdout.decode("utf-8").strip()

        # Se il modello restituisce anche il prompt, prendi solo la parte dopo 📌 **Risposta:**
        marker = "📌 **Risposta:**"
        return raw_output.split(marker)[-1].strip() if marker in raw_output else raw_output

    except subprocess.CalledProcessError as e:
        return f"⚠️ Errore nella generazione della risposta: {e.stderr.decode(errors='ignore')}"
