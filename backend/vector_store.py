# backend/vector_store.py
import chromadb
import os
from chromadb.utils import embedding_functions

# --- ChromaDB Setup --- 
# Using a persistent client to store data on disk
CHROMA_DATA_PATH = "chroma_db"
COLLECTION_NAME = "document_embeddings"

# Ensure the persistence directory exists
os.makedirs(CHROMA_DATA_PATH, exist_ok=True)

# Initialize ChromaDB client
client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

# Get or create the collection. 
# We don't specify an embedding function here if we plan to add pre-computed embeddings.
# If you want ChromaDB to *generate* embeddings, you'd set one, e.g.:
# openai_ef = embedding_functions.OpenAIEmbeddingFunction(
#                 api_key=os.environ.get('OPENAI_API_KEY'), 
#                 model_name="text-embedding-ada-002"
#             )
# collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=openai_ef)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

def add_embeddings(texts: list[str], embeddings: list[list[float]], metadatas: list[dict]):
    """Add text chunks, pre-computed embeddings, and metadata to the Chroma collection."""
    if not texts or not embeddings or not metadatas:
        print("Error: Empty lists provided to add_embeddings.")
        return
    if len(texts) != len(embeddings) or len(texts) != len(metadatas):
        print(f"Error: Mismatched lengths in add_embeddings. Texts: {len(texts)}, Embeddings: {len(embeddings)}, Metadatas: {len(metadatas)}")
        return

    # Generate unique IDs for each chunk (required by ChromaDB)
    # Using filename and chunk index for uniqueness
    ids = [f"{meta.get('filename', 'unknown')}_chunk{meta.get('chunk_index', i)}" for i, meta in enumerate(metadatas)]

    try:
        print(f"Adding {len(texts)} embeddings to Chroma collection '{COLLECTION_NAME}'.")
        collection.add(
            embeddings=embeddings,
            documents=texts, # Store the original text chunk
            metadatas=metadatas,
            ids=ids
        )
        print(f"Successfully added embeddings. Collection count: {collection.count()}")
    except Exception as e:
        print(f"Error adding embeddings to ChromaDB: {e}")
        # Consider re-raising or more specific error handling

def find_similar_chunks(query_embedding: list[float], top_k: int = 3) -> list[str]:
    """Find the most similar text chunks in ChromaDB based on the query embedding."""
    if collection.count() == 0:
        print("Vector store is empty. Cannot perform similarity search.")
        return []
    if not query_embedding:
        print("Error: No query embedding provided.")
        return []
        
    try:
        print(f"Querying Chroma collection '{COLLECTION_NAME}' for {top_k} nearest neighbors.")
        # Query the collection using the pre-computed query embedding
        results = collection.query(
            query_embeddings=[query_embedding], # Note: query_embeddings expects a list of embeddings
            n_results=top_k,
            include=['documents'] # We only need the text content for context
        )
        
        # Extract the document texts from the results
        # Results is a dict containing lists for ids, embeddings, documents, metadatas, distances
        # We access the first (and only) list of documents corresponding to our single query embedding
        similar_docs = results.get('documents', [[]])[0]
        print(f"Found {len(similar_docs)} similar document chunks.")
        return similar_docs
    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        return [] # Return empty list on error

def clear_vector_store():
    """Delete and recreate the Chroma collection to clear the data."""
    global collection
    try:
        print(f"Clearing vector store by deleting collection: {COLLECTION_NAME}")
        client.delete_collection(name=COLLECTION_NAME)
        # Recreate the collection immediately after deletion
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' cleared and recreated. Count: {collection.count()}")
    except Exception as e:
        print(f"Error clearing ChromaDB collection: {e}") 