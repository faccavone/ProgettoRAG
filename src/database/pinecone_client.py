import pinecone
from config.settings import PINECONE_API_KEY, PINECONE_INDEX_NAME

# Inizializza Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

if PINECONE_INDEX_NAME not in [idx["name"] for idx in pc.list_indexes()]:
    pc.create_index(PINECONE_INDEX_NAME, dimension=384, metric="cosine")
    
# Connessione all'indice
index = pc.Index(PINECONE_INDEX_NAME)
