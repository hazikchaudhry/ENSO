# AI-Powered Learning Assistant

A modern PyQt6 application that uses the Feynman Technique and RAG (Retrieval-Augmented Generation) to create an interactive learning experience. The AI guides users through topics using Socratic dialogue, helping them achieve deeper understanding through progressive questioning.

## Features

- 🎨 Modern gradient UI with smooth transitions
- 📚 Support for PDF, DOCX, and TXT documents
- 📑 Page range selection for focused learning
- 🤖 Multiple AI model options (deepseek-r1:7b, mistral, llama2, codellama)
- 💡 Intelligent RAG-powered responses
- 🎯 Socratic teaching method with progressive hints
- 💬 Rich text formatting in chat

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install Ollama and download required models:
```bash
# Install Ollama (Mac/Linux)
curl https://ollama.ai/install.sh | sh

# Pull required models
ollama pull deepseek-r1:7b
ollama pull mistral
```

3. Run the application:
```bash
python main.py
```

## Usage

1. Launch the application
2. Click "Upload Document" to select your learning material (PDF/DOCX/TXT)
3. Select the page range you want to focus on
4. Choose your preferred AI model
5. Click "Start Learning" to begin
6. Engage in a Socratic dialogue with the AI to deepen your understanding

## How It Works

The application uses:
- FAISS for efficient vector similarity search
- LangChain for RAG implementation
- Ollama for local LLM inference
- PyQt6 for the modern UI
- Document processing libraries for text extraction

The AI follows the Feynman Technique principles:
1. Asks you to explain concepts in your own words
2. Identifies gaps in understanding
3. Provides progressive hints rather than direct answers
4. Encourages connections between concepts
5. Uses analogies and real-world examples
