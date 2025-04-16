from config.settings import HUGGINGFACE_API_KEY
import aiohttp  # Per gestire richieste asincrone HTTP
import asyncio  # Per la gestione dell'asincrono

# 📌 Funzione per generare la risposta usando il modello Hugging Face (con un approccio asincrono)
async def generate_answer_async(context, question):
    API_URL = "http://localhost:1234/v1/completions"  # LM Studio endpoint

    # 📌 Costruzione del prompt per il modello Hugging Face
    prompt = f"""
    Sei un assistente esperto di recupero aumentato di informazioni (RAG), specializzato nell'elaborare risposte basate esclusivamente sul contesto fornito.

    🎯 Istruzioni:
    - Leggi con attenzione tutti i documenti del contesto.
    - Rispondi in modo chiaro, coerente e ben strutturato.
    - NON ripetere contenuti simili o identici.
    - NON aggiungere contenuti esterni al contesto.
    - Se le informazioni nel contesto sono insufficienti, dillo chiaramente.

    📌 **Contesto:**  
    {context}  

    📌 **Domanda:**  
    {question}  

    📌 **Risposta:**  
    """

    payload = {
    "model": "llama-3.2-1b-instruct",
    "prompt": prompt,
    "max_tokens": 600,  
    "temperature": 0.3,
    "top_p": 0.5
    }


    async with aiohttp.ClientSession() as session:
        async with session.post(API_URL, json=payload) as response:
            response_data = await response.json()  # Attende la risposta dal modello

            # 📌 Verifica se la risposta contiene una risposta valida
            if isinstance(response_data, list) and len(response_data) > 0 and "generated_text" in response_data[0]:
                raw_text = response_data[0]["generated_text"]
            elif isinstance(response_data, dict) and "generated_text" in response_data:
                raw_text = response_data["generated_text"]
            else:
                return "⚠️ Errore: la risposta dell'API non contiene una risposta valida."

            # 📌 Pulisce la risposta rimuovendo l'intestazione
            cleaned_text = raw_text.split("📌 **Risposta:**")[1].strip() if "📌 **Risposta:**" in raw_text else raw_text
            return cleaned_text

# 📌 Funzione per avviare la chiamata asincrona senza bloccare il flusso
def async_generate_answer(context, question):
    return asyncio.run(generate_answer_async(context, question))  # Usa asyncio.run() per eseguire la funzione asincrona