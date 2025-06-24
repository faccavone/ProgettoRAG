import os
import chromadb

# Usa la funzione PersistentClient per una creazione più chiara e robusta del client persistente
persist_path = os.path.abspath("./chroma_db")
print(f"Persist directory assoluta: {persist_path}")

client = chromadb.PersistentClient(path=persist_path)

# Crea o ottieni la collezione
collection = client.get_or_create_collection("my_collection")

# Aggiungi dati
collection.add(
    documents=["hello world"],
    embeddings=[[0.1] * 1536],
    metadatas=[{"source": "test"}],
    ids=["doc1"]
)

print("Dati aggiunti. Controlla la cartella chroma_db.")

# Controlla se la cartella persiste
print("Cartella esiste dopo?", os.path.exists(persist_path))

# Prova a leggere i dati per verifica
results = collection.get(ids=["doc1"])
print("Dati recuperati:", results)
