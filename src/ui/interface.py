import gradio as gr
from retrieval.search import search_documents  # Corretto import per la ricerca
from AI.response_generator import generate_answer  # Corretto percorso

# 📌 Funzione per la UI
def chatbot_interface(question):
    retrieved_docs = search_documents(question)
    context = " ".join([doc[0] for doc in retrieved_docs])  # Unisce i documenti trovati
    answer = generate_answer(context, question)
    return answer

# 📌 Configura Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🔍 RAG Chatbot")
    question_input = gr.Textbox(label="Inserisci la tua domanda")
    answer_output = gr.Textbox(label="Risposta generata", interactive=False)
    ask_button = gr.Button("Chiedi")
    ask_button.click(chatbot_interface, inputs=question_input, outputs=answer_output)

# 📌 Avvio UI
def start_interface():
    demo.launch()