import chromadb
from chromadb.config import Settings

persist_path = "./chroma_db"

client = chromadb.PersistentClient(path=persist_path)

collection = client.get_or_create_collection("my_collection")
