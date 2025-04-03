# backend/main.py
import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Annotated
from openai import AuthenticationError, RateLimitError, APIError

# Import implemented functions from other modules
from processing import (
    parse_document,
    chunk_text,
    generate_embeddings,
    generate_chat_response,
    SUPPORTED_FILE_TYPES
)
from vector_store import (
    add_embeddings,
    find_similar_chunks,
    clear_vector_store
)

app = FastAPI()

# --- CORS Configuration --- 
# Allow requests from your frontend development server
# In production, restrict this to your actual frontend URL
origins = [
    "http://localhost:5173", # Default Vite dev server port
    "http://127.0.0.1:5173",
    # Add other origins if needed (e.g., your deployed frontend URL)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Directory for uploads ---
UPLOAD_DIR = "uploaded_files"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# --- API Key Management --- 
async def get_api_key(x_api_key: Annotated[str | None, Header()] = None):
    if not x_api_key:
        raise HTTPException(status_code=401, detail="X-API-Key header missing") # 401 Unauthorized
    # Basic check - could add more validation
    if not x_api_key.startswith("sk-"): # Example: Simple check for OpenAI key format
         print(f"Warning: API Key does not look like a standard OpenAI key: {x_api_key[:5]}...")
         # raise HTTPException(status_code=400, detail="Invalid API Key format")
    return x_api_key

# --- Pydantic Models --- 
class ChatQuery(BaseModel):
    query: str

class UploadResponse(BaseModel):
    filename: str
    message: str
    chunk_count: int
    vector_count: int # Number of vectors added to store

class ChatResponse(BaseModel):
    response: str

# --- API Endpoints --- 

@app.post("/upload/", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...), api_key: str = Depends(get_api_key)):
    """Endpoint to upload a document, process it, and store embeddings."""
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in SUPPORTED_FILE_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}. Supported types: {SUPPORTED_FILE_TYPES}")

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        # Save the uploaded file temporarily
        print(f"Saving uploaded file to: {file_path}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 1. Parse Document
        text_content = parse_document(file_path, file_extension)
        if not text_content:
             # Specific error from parsing is logged in parse_document
            raise HTTPException(status_code=422, detail="Failed to parse document content. Check file format and content.")

        # 2. Chunk Text
        chunks = chunk_text(text_content)
        if not chunks:
            raise HTTPException(status_code=422, detail="Failed to chunk document text. Document might be empty or unprocessable.")

        # 3. Generate Embeddings (using the provided API key)
        embeddings = generate_embeddings(chunks, api_key)
        # generate_embeddings now raises exceptions on failure
        if not embeddings or len(embeddings) != len(chunks):
             # This case should ideally not happen if generate_embeddings raises, but as a safeguard:
             raise HTTPException(status_code=500, detail="Mismatch in generated embeddings count.")

        # 4. Store Embeddings
        # Create metadata including the filename for each chunk
        metadata = [{"filename": file.filename, "chunk_index": i} for i in range(len(chunks))]
        add_embeddings(texts=chunks, embeddings=embeddings, metadatas=metadata)

        return UploadResponse(
            filename=file.filename,
            message="Document processed and embeddings stored successfully.",
            chunk_count=len(chunks),
            vector_count=len(embeddings)
        )

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Uploaded file not found after saving. Check server permissions.")
    except AuthenticationError:
        raise HTTPException(status_code=401, detail="OpenAI Authentication Error: Invalid API Key provided.")
    except RateLimitError:
        raise HTTPException(status_code=429, detail="OpenAI Rate Limit Error: Too many requests. Please try again later.")
    except APIError as e:
         raise HTTPException(status_code=502, detail=f"OpenAI API Error: {e}") # 502 Bad Gateway
    except ValueError as ve: # Catch specific errors like missing API key from processing
        raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException as http_exc:
        # Re-raise HTTPExceptions to let FastAPI handle them
        raise http_exc
    except Exception as e:
        # Catch any other unexpected errors during processing
        print(f"Error processing file {file.filename}: {e}")
        # Consider logging the full traceback here
        raise HTTPException(status_code=500, detail=f"Internal server error processing file: {e}")
    finally:
        # Clean up the temporarily saved file
        if os.path.exists(file_path):
            print(f"Cleaning up temporary file: {file_path}")
            os.remove(file_path)
        # Close the uploaded file object
        if file and not file.file.closed:
             await file.close()

@app.post("/chat/", response_model=ChatResponse)
async def chat_with_documents(query: ChatQuery, api_key: str = Depends(get_api_key)):
    """Endpoint to handle user chat queries."""
    user_query = query.query

    try:
        # 1. Generate embedding for the user query (using the provided API key)
        query_embedding_list = generate_embeddings([user_query], api_key)
        # generate_embeddings raises exceptions on failure
        if not query_embedding_list: # Should be caught by exception, but safeguard
             raise HTTPException(status_code=500, detail="Failed to generate embedding for the query.")
        query_embedding = query_embedding_list[0]

        # 2. Find relevant document chunks from vector store
        # find_similar_chunks now returns a list of document strings
        context_chunks = find_similar_chunks(query_embedding, top_k=3) # Get top 3 most similar chunks

        if not context_chunks:
            print("No relevant document chunks found for the query.")
            # Let the LLM handle the lack of context based on its system prompt
            # ai_response = "I couldn't find relevant information in the uploaded documents to answer your question."

        # 3. Generate response using LLM with context (using the provided API key)
        # This function now returns error messages directly on failure
        ai_response = generate_chat_response(user_query, context_chunks, api_key)

        # Check if the response indicates an internal error (starts with "Error:")
        if ai_response.startswith("Error:"):
            # Map specific internal errors to HTTP status codes if needed,
            # otherwise return a generic server error. Currently returns 503 Service Unavailable.
             if "authenticate" in ai_response.lower():
                 raise HTTPException(status_code=401, detail=ai_response)
             elif "rate limit" in ai_response.lower():
                 raise HTTPException(status_code=429, detail=ai_response)
             else:
                raise HTTPException(status_code=503, detail=ai_response) # Service Unavailable

        return ChatResponse(response=ai_response)

    except AuthenticationError:
        raise HTTPException(status_code=401, detail="OpenAI Authentication Error: Invalid API Key provided.")
    except RateLimitError:
        raise HTTPException(status_code=429, detail="OpenAI Rate Limit Error: Too many requests. Please try again later.")
    except APIError as e:
         raise HTTPException(status_code=502, detail=f"OpenAI API Error: {e}")
    except ValueError as ve: # Catch specific errors like missing API key
        raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Error during chat processing for query '{user_query}': {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during chat: {e}")


@app.post("/clear/")
async def clear_data():
    """Endpoint to clear the ChromaDB vector store collection."""
    try:
        clear_vector_store()
        # Optionally, also clear the uploaded_files directory if desired
        # Be careful with this in production!
        # upload_dir_path = UPLOAD_DIR
        # if os.path.exists(upload_dir_path):
        #     print(f"Also removing directory: {upload_dir_path}")
        #     shutil.rmtree(upload_dir_path)
        #     os.makedirs(upload_dir_path)
        return {"message": "Vector store collection cleared successfully."}
    except Exception as e:
        print(f"Error clearing data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear vector store: {e}")

# --- Run the application (for local development) ---
if __name__ == "__main__":
    import uvicorn
    print(f"Starting Uvicorn server on http://127.0.0.1:8000")
    # Ensure ChromaDB collection is loaded/created before starting server
    print(f"ChromaDB collection '{vector_store.COLLECTION_NAME}' count: {vector_store.collection.count()}")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True) 