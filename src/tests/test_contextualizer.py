import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_processing.pdf_loader import read_pdfs_from_folder
from data_processing.text_chunker import chunk_text
from data_processing.contextualizer import generate_chunk_context
from config.settings import CHUNK_SIZE, OVERLAP, PDF_FOLDER


# Caricamento PDF per test
print("📄 Caricamento PDF per test...")

pdfs = read_pdfs_from_folder(PDF_FOLDER)

if not pdfs:
    print("❌ Nessun PDF trovato nella cartella.")
else:
    # Solo il primo PDF
    name, text = pdfs[0]
    print(f"✅ Usando PDF: {name}")

    # Primi 2 chunk
    chunks = chunk_text(text, CHUNK_SIZE, OVERLAP)[:2]

    for i, chunk in enumerate(chunks):
        print(f"\n=== CHUNK {i + 1} ===")
        
        # Genera contesto
        context = generate_chunk_context(text, chunk)
        
        # Mostra contesto separato
        print(f"\n🔹 Context:\n{context}")

        # Mostra chunk separato
        print(f"\n🔸 Original Chunk:\n{chunk}")

        # Mostra il risultato finale
        contextualized_chunk = f"{context}\n\n{chunk}"
        print(f"\n🧠 Contextualized Chunk:\n{contextualized_chunk}")
