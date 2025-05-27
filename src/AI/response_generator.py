import google.generativeai as genai
from config.settings import GEMINI_API_KEY

# ✅ Configura Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")  

def generate_answer(context: str, question: str) -> str:
    """
    Usa Gemini per generare una risposta basata sul contesto fornito.
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
        response = model.generate_content(
            prompt.strip(),
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 1000
            }
        )
        answer = response.text.strip()

        # Rimuove eventualmente il marcatore finale
        marker = "📌 **Risposta:**"
        return answer.split(marker)[-1].strip() if marker in answer else answer

    except Exception as e:
        return f"⚠️ Errore durante la generazione con Gemini: {e}"
