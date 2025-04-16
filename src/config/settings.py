# 📌 Configura API Keys e parametri
HUGGINGFACE_API_KEY = "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"
PINECONE_API_KEY = "pcsk_6NrHUY_EQgh4FJkuQk3wLTeZJ4vR3W2R1Y5HdScizXDTHY5GhUKdJEGT2syVjK72cVyspD"
PINECONE_INDEX_NAME = "rag-demo"
CHUNK_SIZE = 300
OVERLAP = 50
INITIAL_K = 100 # Numero iniziale di documenti recuperati
TOP_K = 20  # Numero finale di documenti recuperati
SIMILARITY_THRESHOLD = 0.4
BM25_INDEX_PATH = "bm25_index.pkl"  # File dove salviamo l'indice BM25
PDF_FOLDER = "data" #Cartella con i documenti da leggere