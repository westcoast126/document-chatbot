# Document Chatbot Application

This project provides a basic structure for a web application that allows users to chat with an AI based on the content of uploaded documents.

## Project Structure

```
.
├── backend/
│   ├── uploaded_files/      # Temporarily stores uploaded files (should be gitignored)
│   ├── main.py              # FastAPI application, API endpoints
│   ├── processing.py        # Placeholder functions for document parsing, chunking, embeddings
│   ├── vector_store.py      # Placeholder for vector storage and similarity search
│   └── requirements.txt     # Python backend dependencies
├── frontend/
│   ├── public/              # Static assets
│   ├── src/
│   │   ├── components/      # React components
│   │   │   ├── ApiKeyInput.tsx
│   │   │   ├── ChatInterface.tsx
│   │   │   └── FileUpload.tsx
│   │   ├── App.tsx          # Main application component
│   │   ├── index.css        # Main CSS file (includes Tailwind)
│   │   └── main.tsx         # Entry point for React app
│   ├── index.html           # Main HTML page
│   ├── package.json         # Frontend dependencies and scripts
│   ├── tailwind.config.js   # Tailwind CSS configuration
│   ├── postcss.config.js    # PostCSS configuration
│   └── tsconfig.json        # TypeScript configuration
└── README.md                # This file
```

## Setup and Running

**Prerequisites:**

*   Node.js and npm (for frontend)
*   Python 3.8+ and pip (for backend)

**1. Backend Setup:**

```bash
# Navigate to the backend directory
cd backend

# Create a virtual environment (recommended)
python -m venv venv
# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the backend server (FastAPI with Uvicorn)
uvicorn main:app --reload
# The backend should now be running at http://127.0.0.1:8000
```

**2. Frontend Setup:**

```bash
# Navigate to the frontend directory (from the root)
cd frontend

# Install dependencies (if you haven't already or encounter issues)
npm install

# Run the frontend development server (Vite)
npm run dev
# The frontend should now be running at http://localhost:5173 (or another port if 5173 is busy)
```

**3. Using the Application:**

*   Open your browser to the frontend URL (e.g., `http://localhost:5173`).
*   Enter your API key (e.g., from OpenAI) in the sidebar.
*   Upload a document (`.txt`, `.pdf`, `.md`).
*   Ask questions related to the document content in the chat interface.

## Next Steps & Implementation Details

*   **Backend `processing.py`:** Implement actual document parsing (e.g., using `pypdf`), text chunking (e.g., using `langchain.text_splitter`), and embedding generation (e.g., using the `openai` or `anthropic` Python libraries with the user's API key).
*   **Backend `vector_store.py`:** Replace the in-memory store with a persistent vector database like ChromaDB or FAISS for efficient storage and similarity search.
*   **Backend `main.py`:** Enhance error handling, add API key validation, and potentially implement asynchronous processing for large documents.
*   **Frontend:** Improve UI/UX, add loading states, handle errors more gracefully, potentially add document management features (list, delete), and consider full Markdown rendering for chat messages.
*   **Security:** Implement more robust API key handling (e.g., environment variables on the server, never storing directly in frontend code). Add authentication if needed.
*   **Deployment:** Configure CORS properly for production, containerize the application (e.g., using Docker), and deploy frontend and backend services. 