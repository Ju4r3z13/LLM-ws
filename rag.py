import chromadb
from ollama import Client

# Connect to Ollama
ollama = Client(host="http://localhost:11434")

# Create/load persistent ChromaDB database
chroma = chromadb.PersistentClient(path="./chroma_db")

# Create or load a collection
collection = chroma.get_or_create_collection(
    name="" #Collection
)

# Read document
with open("", "r") as file: #Document to be read
    text = file.read()

# Generate embedding
response = ollama.embed(
    model="qwen3-embedding:0.6b",
    input=text
)

embedding = response["embeddings"][0]

# Store document in ChromaDB
collection.upsert(
    ids=[], #ID
    documents=[text],
    embeddings=[embedding]
)

print("Document stored successfully!")
print(f"Embedding dimensions: {len(embedding)}")
