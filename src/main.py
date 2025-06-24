import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
from retrieval.search import search_documents, search_relevant_images
from AI.response_generator import generate_answer

import warnings
from urllib3.exceptions import NotOpenSSLWarning

warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

def chatbot_interface(question):
    print(f"🟡 DOMANDA RICEVUTA: {question}")

    # 🔹 Fase 1: Recupera contenuti testuali + descrizioni immagini
    retrieved_docs = search_documents(question)

    if isinstance(retrieved_docs, list) and len(retrieved_docs) == 1 and isinstance(retrieved_docs[0], str):
        print("⚠️ Risposta breve, nessuna elaborazione.")
        return retrieved_docs[0], []

    # 🔹 Costruzione contesto
    context_chunks = [doc["text"] for doc in retrieved_docs]
    context = " ".join(context_chunks)

    # 🔹 Genera la risposta
    answer = generate_answer(context, question)

    # 🔹 Seconda fase: recupero immagini con query + risposta
    refined_query = question + " " + answer
    image_results = search_relevant_images(refined_query, top_k=10)

    # 🔹 Filtra immagini valide
    image_paths = []
    for doc in image_results:
        path = doc.get("image_path")
        if path:
            exists = os.path.exists(path)
            if exists:
                image_paths.append(path)

    return answer, image_paths

def main():
    st.set_page_config(page_title="RAG Chatbot con Immagini", layout="wide")
    st.title("🔍 RAG Chatbot con immagini")

    question = st.text_input("Inserisci la tua domanda:")

    if st.button("Chiedi") and question.strip() != "":
        with st.spinner("💡 Elaborazione in corso..."):
            answer, images = chatbot_interface(question)

        st.markdown("### 💬 Risposta generata:")
        st.write(answer)

        if images:
            st.markdown("### 🖼️ Immagini correlate alla risposta:")
            cols = st.columns(len(images))
            for col, img_path in zip(cols, images):
                col.image(img_path, use_container_width=True)
        else:
            st.info("Nessuna immagine trovata pertinente alla risposta.")

if __name__ == "__main__":
    main()
