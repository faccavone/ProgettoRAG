from config.settings import HUGGINGFACE_API_KEY
import aiohttp  # Per gestire richieste HTTP in modo asincrono
import asyncio  # Per gestire le coroutine asincrone

# 📌 Funzione asincrona che invia una richiesta al modello locale (via LM Studio) per generare una risposta basata su un contesto
async def generate_answer_async(context, question):
    API_URL = "http://localhost:1234/v1/completions"  # Endpoint del modello LM Studio

    # Prompt dettagliato che istruisce il modello su come comportarsi durante la generazione della risposta
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

    # Corpo della richiesta HTTP: specifica il modello, prompt e parametri di generazione
    payload = {
        "model": "llama-3.2-1b-instruct",  # Nome del modello usato (compatibile con LM Studio)
        "prompt": prompt,
        "max_tokens": 600,  # Numero massimo di token nella risposta
        "temperature": 0.3,  # Controlla la creatività della risposta
        "top_p": 0.5  # Probabilità cumulativa per filtrare token
    }

    # Avvia una sessione HTTP asincrona per inviare la richiesta POST al modello
    async with aiohttp.ClientSession() as session:
        async with session.post(API_URL, json=payload) as response:
            response_data = await response.json()  # Attende e converte la risposta JSON

            # 📌 Estrae la risposta generata (se disponibile) in base al formato restituito
            if isinstance(response_data, list) and len(response_data) > 0 and "generated_text" in response_data[0]:
                raw_text = response_data[0]["generated_text"]
            elif isinstance(response_data, dict) and "generated_text" in response_data:
                raw_text = response_data["generated_text"]
            else:
                return "⚠️ Errore: la risposta dell'API non contiene una risposta valida."

            # Rimuove eventuali duplicazioni o intestazioni nel testo restituito
            cleaned_text = raw_text.split("📌 **Risposta:**")[1].strip() if "📌 **Risposta:**" in raw_text else raw_text
            return cleaned_text

# 📌 Funzione sincrona che lancia la coroutine asincrona per ottenere una risposta
def async_generate_answer(context, question):
    return asyncio.run(generate_answer_async(context, question))  # Esegue la funzione asincrona con il loop asyncio
