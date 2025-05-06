import sys
import os
import time

# Per importare i moduli dalle cartelle superiori
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_processing.pdf_loader import read_pdfs_from_folder
from data_processing.text_chunker import chunk_text
from data_processing.contextualizer import generate_chunk_context, build_chunk_context_window
from config.settings import CHUNK_SIZE, OVERLAP, PDF_FOLDER


# Funzione per gestire il contesto dei chunk con la pausa tra le chiamate
def process_chunks_with_pause(chunks: list[str], window_size: int = 3, chunk_limit_per_minute: int = 4):
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


print("📄 Caricamento PDF per test...")

# Carica i PDF dalla cartella specificata
pdfs = read_pdfs_from_folder(PDF_FOLDER)

if not pdfs:
    print("❌ Nessun PDF trovato nella cartella.")
else:
    # Usare un singolo PDF di esempio
    name, text = pdfs[0]
    print(f"✅ Usando PDF: {name}")

    # Generazione dei chunk dal testo
    chunks = chunk_text(text, CHUNK_SIZE, OVERLAP)
    print("Numero di chunk creati:", len(chunks))
    # Test sui primi 5 chunk con la nuova funzione che gestisce la pausa
    processed_results = process_chunks_with_pause(chunks)

    # Visualizzazione dei risultati
    for i, (chunk, context) in enumerate(processed_results):
        print(f"\n=== CHUNK {i + 1} ===")
        print(f"\n🔹 Context:\n{context}")
        print(f"\n🔸 Original Chunk:\n{chunk}")
