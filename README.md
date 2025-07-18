# Enso - AI-Powered Learning Assistant

<img width="1247" alt="ENSO" src="https://github.com/user-attachments/assets/50808cf1-891c-4dc2-bbf7-f827b8fa3f07" />

Enso is a modern PyQt6 application that implements an Local LLM-powered learning companion using the Feynman Technique. The application processes documents using natural language processing and engages users in a conversational learning experience guided by an AI model.

## Features

- 🎨 Modern gradient UI with smooth transitions and ChatGPT-like interface
- 📚 Support for PDF, DOCX, and TXT documents
- 📑 Page range selection for focused learning
- 🤖 Local model integration for intelligent responses
- 💡 RAG-powered context retrieval using FAISS
- 🎯 Socratic teaching method with progressive hints
- 💬 Rich text formatting with bold, italic, and bullet points
- ⚡ Real-time token streaming for dynamic responses

## Setup Instructions For Electron

1. Clone the repository:
```bash
git clone https://github.com/hazikchaudhry/ENSO.git
cd ENSO
```

2. Install dependencies:
```bash
cd electron-app
npm install
```

3. Install Python dependencies:
```bash
cd backend
pip install -r requirements.txt
```

4. Run the application:
```bash
# Terminal 1 - Start backend
cd backend
python server.py

# Terminal 2 - Start Electron app
cd electron-app
npm start
```

## Setup For AI

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install Ollama on Windows at https://ollama.com/ and download the Mistral model OR Gemma model:
```bash
# Pull required model
ollama pull mistral:7b-instruct
ollama run gemma3:4b
```

3. Run the application:
```bash
python main.py
```

## Usage

1. Launch the application
2. Click "Upload Document" to select your learning material (PDF/DOCX/TXT)
3. Select the page range you want to focus on
4. Verify Ollama is running locally
5. Start your learning journey with the AI companion

## Technical Architecture

The application leverages several key technologies:

- **Document Processing**: PyMuPDF (fitz) and python-docx for text extraction
- **AI/ML Components**: 
  - LangChain for orchestrating AI workflows
  - Ollama for local LLM inference
  - HuggingFace embeddings for text vectorization
  - FAISS for efficient vector similarity search
- **UI Framework**: PyQt6 with modern gradient styling

## Learning Approach

Enso implements the Feynman Technique through:
1. Engaging users in natural dialogue about concepts
2. Identifying knowledge gaps through Socratic questioning
3. Providing progressive hints rather than direct answers
4. Using analogies and real-world examples
5. Encouraging users to explain concepts in their own words

## Features in Detail

- **Smart Context Retrieval**: Uses FAISS similarity search to find relevant context
- **Real-time Responses**: Streams tokens for dynamic response generation
- **Rich Text Support**: Format your messages with bold, italic, and bullet points
- **Conversation Management**: Tracks conversation state and maintains context
- **Error Handling**: Robust error management with user-friendly messages
