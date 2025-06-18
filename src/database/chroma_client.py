import chromadb
from chromadb.config import Settings

client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_db"  # Cartella per la persistenza locale
))

collection = client.get_or_create_collection("rag_chunks")
