import gradio as gr
from retrieval.search import search_documents
from AI.response_generator import generate_answer
from PIL import Image
import os

# 📌 Funzione principale per la UI
def chatbot_interface(question):
    retrieved_docs = search_documents(question)

    # 🔹 Crea il contesto testuale (solo i testi dei risultati)
    context_chunks = [doc["text"] for doc in retrieved_docs]
    context = " ".join(context_chunks)

    # 🔹 Genera la risposta usando il contesto
    answer = generate_answer(context, question)

    # 🔹 Estrai immagini dai risultati se presenti
    image_paths = [
        doc["image_path"] for doc in retrieved_docs
        if doc["type"] == "image" and doc.get("image_path") and os.path.exists(doc["image_path"])
    ]

    return answer, image_paths

# 📌 Costruzione interfaccia Gradio
with gr.Blocks() as demo:
    gr.Markdown("# 🔍 RAG Chatbot con immagini recuperate")

    # Campo input domanda
    question_input = gr.Textbox(label="Inserisci la tua domanda")

    # Campo output risposta generata
    answer_output = gr.Textbox(label="Risposta generata", interactive=False)

    # Sezione immagini, racchiusa in un Accordion (collassabile)
    with gr.Accordion("📸 Immagini recuperate", open=False):
        image_gallery = gr.Gallery(
            show_label=False,
            visible=True,
            height=300,
            columns=3,
            object_fit="contain"
        )

    # Pulsante invio domanda
    ask_button = gr.Button("Chiedi")

    # Collegamento funzione → UI
    ask_button.click(
        fn=chatbot_interface,
        inputs=question_input,
        outputs=[answer_output, image_gallery]
    )

# 📌 Avvia la demo
def start_interface():
    demo.launch()
