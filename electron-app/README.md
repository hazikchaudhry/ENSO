# Enso - AI Learning Assistant (Electron Version)

This is the Electron frontend version of Enso, an AI-powered learning assistant that uses the Feynman Technique to help users understand complex topics through interactive conversation.

## Features

- Document processing (PDF, DOCX, TXT)
- AI-powered interactive learning using the Feynman Technique
- Conversational interface with streaming responses
- Electron-based desktop application with Python backend

## Prerequisites

- Node.js and npm
- Python 3.8 or higher
- Ollama running locally (for AI model access)

## Setup

### 1. Install Python Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Install Node.js Dependencies

```bash
npm install
```

### 3. Start Ollama

Ensure Ollama is running locally with the required models (e.g., mistral:7b-instruct).

## Running the Application

To run both the Python backend and Electron frontend together:

```bash
npm run dev
```

Or start them separately:

```bash
# Start the backend
npm run start-backend

# In another terminal, start the frontend
npm run start
```

## How to Use

1. **Upload a Document**:
   - Click the "Upload" button to select a document (PDF, DOCX, or TXT)
   - Optionally specify page ranges for PDF documents
   - Select the AI model to use
   - Wait for processing to complete

2. **Chat with Enso**:
   - Once processing is complete, the document section will collapse
   - Type your questions or comments in the input area
   - Press Enter or click the send button to submit
   - Enso will respond with questions to help deepen your understanding

3. **Text Formatting**:
   - Use the B button for bold text
   - Use the I button for italic text
   - Use the • button to add bullet points

## Architecture

This application consists of two main components:

1. **Python Backend** (Flask server):
   - Document processing (text extraction, chunking)
   - Vector embeddings creation and storage
   - AI model integration via Ollama
   - Chat response generation

2. **Electron Frontend**:
   - User interface for document upload and chat
   - Communication with backend via HTTP API
   - File system access for document selection
   - Streaming response display

## API Endpoints

The backend exposes the following API endpoints:

- `POST /process` - Start document processing
- `GET /status` - Check processing status
- `POST /chat` - Send a message and get a response