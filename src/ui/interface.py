import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from retrieval.search import search_documents
from AI.response_generator import generate_answer

def chatbot_interface(question):
    retrieved_docs = search_documents(question)

    # Caso errore testuale (es. query troppo breve)
    if isinstance(retrieved_docs, list) and len(retrieved_docs) == 1 and isinstance(retrieved_docs[0], str):
        return retrieved_docs[0], []

    # Costruzione contesto da chunk testuali + descrizioni immagine
    context_chunks = [doc["text"] for doc in retrieved_docs]
    context = " ".join(context_chunks)

    # Risposta generata dal LLM
    answer = generate_answer(context, question)

    # Seleziona immagini valide da visualizzare (fino a 3)
    image_paths = [
        doc.get("image_path") for doc in retrieved_docs
        if doc.get("type") == "image" and doc.get("image_path") and os.path.exists(doc["image_path"])
    ]

    return answer, image_paths[:3]

def main():
    st.title("🔍 RAG Chatbot con immagini recuperate")

    question = st.text_input("Inserisci la tua domanda")

    if st.button("Chiedi") and question.strip() != "":
        answer, images = chatbot_interface(question)

        st.markdown("### Risposta generata:")
        st.write(answer)

        if images:
            st.markdown("### Immagini recuperate:")
            cols = st.columns(len(images))
            for col, img_path in zip(cols, images):
                col.image(img_path, use_column_width=True)
        else:
            st.write("Nessuna immagine trovata.")

if __name__ == "__main__":
    main()
