# backend/processing.py

# This module will handle document parsing, text chunking, and embedding generation.

# --- Placeholder Functions --- 
# Replace these with actual implementations using libraries like
# pypdf, langchain, openai, etc.

import os
from openai import OpenAI, AuthenticationError, RateLimitError, APIError
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

SUPPORTED_FILE_TYPES = {".txt", ".pdf", ".md"}
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small" # Consider making this configurable
DEFAULT_CHAT_MODEL = "gpt-3.5-turbo"      # Consider making this configurable

def parse_document(file_path: str, file_type: str) -> str:
    """Parses text content from a supported file."""
    print(f"Parsing document: {file_path} ({file_type})")
    text = ""
    try:
        if file_type == ".txt" or file_type == ".md":
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        elif file_type == ".pdf":
            reader = PdfReader(file_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n" # Add newline between pages
        else:
            print(f"Unsupported file type: {file_type}")
            # No exception, just return empty string for unsupported types

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        raise # Re-raise FileNotFoundError to be handled by the endpoint
    except Exception as e:
        print(f"Error parsing {file_type} file {file_path}: {e}")
        # Depending on the error, you might want to return partial text or raise
        # For now, return empty string for general parsing errors
        return ""
    
    print(f"Successfully parsed document. Text length: {len(text)}")
    return text

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> list[str]:
    """Splits text into manageable chunks using LangChain."""
    if not text:
        print("Warning: chunk_text received empty text.")
        return []
        
    print(f"Chunking text (length: {len(text)}), size={chunk_size}, overlap={chunk_overlap}")
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True, # Adds metadata about chunk start index
        )
        # LangChain's splitter returns Document objects, we extract the text content
        docs = text_splitter.create_documents([text])
        chunks = [doc.page_content for doc in docs]
        print(f"Created {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        print(f"Error chunking text: {e}")
        # Fallback or re-raise? For now, return empty list.
        return []

def generate_embeddings(chunks: list[str], api_key: str, model: str = DEFAULT_EMBEDDING_MODEL):
    """Generates embeddings for text chunks using the OpenAI API."""
    print(f"Generating embeddings for {len(chunks)} chunks using model '{model}'.")
    if not api_key:
        print("API Key not provided. Cannot generate embeddings.")
        # Raising an error might be better for explicit failure
        raise ValueError("API Key is required for generating embeddings.")
    if not chunks:
        print("Warning: generate_embeddings received empty list of chunks.")
        return []

    try:
        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(
            input=chunks,
            model=model
        )
        embeddings = [item.embedding for item in response.data]
        print(f"Successfully generated {len(embeddings)} embeddings.")
        return embeddings
    except AuthenticationError:
        print("OpenAI Authentication Error: Invalid API Key?")
        raise # Re-raise to be caught by the endpoint
    except RateLimitError:
        print("OpenAI Rate Limit Error: Please check your usage plan and limits.")
        raise
    except APIError as e:
        print(f"OpenAI API Error: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred during embedding generation: {e}")
        raise # Re-raise generic exceptions

def generate_chat_response(query: str, context_chunks: list[str], api_key: str, model: str = DEFAULT_CHAT_MODEL):
    """Generates a chat response using an OpenAI chat model with context."""
    print(f"Generating chat response for query: '{query}' using model '{model}'.")
    if not api_key:
        print("API Key not provided. Cannot generate chat response.")
        return "Error: API Key is required to generate a response."

    # Construct the prompt with context
    context_str = "\n\n---\n\n".join(context_chunks) # Separate chunks clearly
    system_prompt = "You are a helpful assistant. Answer the user's question based *only* on the provided document excerpts. If the answer is not found in the excerpts, say 'I couldn't find information about that in the provided documents.'"
    user_prompt = f"""Document Excerpts:
---
{context_str}
---

Question: {query}"""

    print(f"-- System Prompt --\n{system_prompt}")
    print(f"-- User Prompt (first 100 chars) --\n{user_prompt[:100]}...")
    
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2, # Lower temperature for more factual answers based on context
        )
        ai_response = response.choices[0].message.content
        print("Successfully generated chat response.")
        return ai_response.strip() if ai_response else "Error: Received an empty response from the language model."

    except AuthenticationError:
        print("OpenAI Authentication Error: Invalid API Key?")
        # Return a user-friendly error message
        return "Error: Could not authenticate with the AI service. Please check your API Key."
    except RateLimitError:
        print("OpenAI Rate Limit Error.")
        return "Error: The AI service rate limit has been exceeded. Please try again later or check your usage plan."
    except APIError as e:
        print(f"OpenAI API Error: {e}")
        return f"Error: An API error occurred while communicating with the AI service: {e}"
    except Exception as e:
        print(f"An unexpected error occurred during chat response generation: {e}")
        return f"Error: An unexpected error occurred while generating the response: {e}" 