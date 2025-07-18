from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
import os
import fitz  # PyMuPDF
from docx import Document
import time
import random
import threading
import json
import argparse

app = Flask(__name__)

# Configure CORS more explicitly with resources setting
CORS(app, resources={r"/*": {
    "origins": "*",  # Allow all origins for now
    "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
    "expose_headers": ["Content-Type", "X-Total-Count"],
    "supports_credentials": True,
    "max_age": 600
}})

# Global variables to store processing state
processing_status = {
    "status": "idle",  # idle, processing, complete, error
    "progress": 0,
    "message": "",
    "error": None
}

# Global variables to store document processing results
vectorstore = None
llm_model = None
conversation_history = []
teaching_state = {
    "current_topic": None,
    "understanding_level": 1,
    "depth_level": 0,
    "related_concepts": []
}

# Stream response handler for chat
class StreamingHandler(BaseCallbackHandler):
    def __init__(self):
        self.tokens = []
        
    def on_llm_new_token(self, token, **kwargs):
        self.tokens.append(token)

def process_document(file_path, start_page, end_page, model_name):
    """Process document in a separate thread"""
    global processing_status, vectorstore, llm_model
    
    try:
        # Update status
        processing_status["status"] = "processing"
        processing_status["message"] = "Extracting text from document..."
        
        # Extract text based on file type
        if file_path.endswith('.pdf'):
            text = process_pdf(file_path, start_page, end_page)
        elif file_path.endswith('.docx'):
            text = process_docx(file_path)
        elif file_path.endswith('.txt'):
            text = process_txt(file_path)
        else:
            
            processing_status["status"] = "error"
            processing_status["message"] = "Unsupported file format"
            return
        
        # Update status
        processing_status["progress"] = 30
        processing_status["message"] = "Creating text chunks..."
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)
        
        # Update status
        processing_status["progress"] = 60
        processing_status["message"] = "Creating embeddings..."
        
        # Create embeddings and vectorstore
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={'device': 'cpu'})
        vectorstore = FAISS.from_texts(chunks, embeddings)
        
        # Update status
        processing_status["progress"] = 80
        processing_status["message"] = "Initializing AI model..."
        
        # Initialize LLM
        llm_model = OllamaLLM(model=model_name, temperature=0.7, streaming=True)
        
        # Update status
        processing_status["status"] = "complete"
        processing_status["progress"] = 100
        processing_status["message"] = "Processing complete!"
        
    except Exception as e:
        processing_status["status"] = "error"
        processing_status["message"] = f"Error: {str(e)}"
        processing_status["error"] = str(e)

def process_pdf(file_path, start_page, end_page):
    """Extract text from PDF within specified page range"""
    text = ""
    with fitz.open(file_path) as doc:
        start = start_page - 1 if start_page else 0
        end = min(end_page if end_page else doc.page_count, doc.page_count)
        for page_num in range(start, end):
            text += doc[page_num].get_text()
    return text

def process_docx(file_path):
    """Extract text from DOCX"""
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def process_txt(file_path):
    """Extract text from TXT"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


@app.before_request
def log_request_info():
    print("==== REQUEST RECEIVED ====")
    print(f"Headers: {request.headers}")
    print(f"Method: {request.method}")
    print(f"Path: {request.path}")
    print(f"Data: {request.get_data()}")
    # Flush to ensure output is visible immediately
    import sys
    sys.stdout.flush()

@app.route('/process', methods=['POST'])
def start_processing():
    """Start document processing"""
    global processing_status
    
    # Reset processing status
    processing_status = {
        "status": "idle",
        "progress": 0,
        "message": "",
        "error": None
    }
    
    # Get request data
    data = request.json
    file_path = data.get('file_path')
    start_page = data.get('start_page', 1)
    end_page = data.get('end_page')
    model_name = data.get('model_name', 'gemma3:4b-it-q4_K_M')
    
    # Validate input
    if not file_path:
        return jsonify({"error": "File path is required"}), 400
    
    if not os.path.exists(file_path):
        return jsonify({"error": "File does not exist"}), 400
    
    # Start processing in a separate thread
    thread = threading.Thread(target=process_document, args=(file_path, start_page, end_page, model_name))
    thread.daemon = True
    thread.start()
    
    return jsonify({"message": "Processing started"})

@app.route('/status', methods=['GET'])
def get_status():
    """Get processing status"""
    return jsonify(processing_status)

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat request"""
    global conversation_history, vectorstore, llm_model
    
    # Check if document has been processed
    if not vectorstore or not llm_model:
        return jsonify({"error": "No document has been processed yet"}), 400
    
    # Get request data
    data = request.json
    user_message = data.get('message')
    
    if not user_message:
        return jsonify({"error": "Message is required"}), 400
    
    # Add to conversation history
    conversation_history.append(user_message)
    
    
    # Get relevant context from vectorstore
    context = ""
    if vectorstore:
        results = vectorstore.similarity_search(user_message, k=2)
        context = "\n".join([doc.page_content for doc in results])
    
    # Format conversation history for better context
    formatted_history = ""
    if len(conversation_history) >= 2:  # Need at least one exchange
        # Get the last 6 messages (3 exchanges)
        recent_history = conversation_history[-6:]
        
        # Format as alternating User/AI messages
        for i in range(0, len(recent_history), 2):
            if i < len(recent_history):
                formatted_history += f"User: {recent_history[i]}\n"
            if i+1 < len(recent_history):
                formatted_history += f"AI: {recent_history[i+1]}\n\n"
    
    # Prepare prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a nonchalant, curious learning companion using the Feynman Technique. Your goal is to have a natural, conversational exchange that guides the user to deeper understanding through thoughtful questions.

CURRENT STATE:
Topic: {topic}
Understanding: Level {understanding} (0: unclear -> 3: excellent)
Depth: Level {depth} (0: basics -> 2: advanced)

CONVERSATION HISTORY:
{history}

CONTEXT (Use to inform questions, never teach directly):
{context}{related_concepts_prompt}

YOUR APPROACH:
1. Be Conversational & Human-like:
   - Use casual, friendly language with occasional emojis (👋, 🤔, 💡)
   - Add conversational elements like "Hmm", "Oh!", "I see", "Interesting!"
   - Use contractions (don't, can't, I'm) and casual phrasing
   - React naturally to what they say with brief acknowledgments
   - Maintain conversation continuity - refer to previous exchanges
   - NEVER introduce yourself again or restart the conversation

2. Varied Question Techniques (Mix these throughout the conversation):
   
   BASIC QUESTIONS (Level 0-1):
   - "So what happens first when...?"
   - "Could you tell me a bit about...?"
   - "What's a simple example of...?"

   METAPHORS & ANALOGIES (Any level):
   - "Is it like [familiar concept]? How so?"
   - "Imagine [concept] is like [everyday object]. Would that work?"
   - "Could we think of this as [simple analogy]?"

   RANKING & COMPARISONS:
   - "Which is more important: X or Y? Why?"
   - "If you had to rank these three factors, how would you order them?"

   SCENARIO-BASED:
   - "What if we tried to [scenario]? What would happen?"
   - "Imagine you're explaining this to someone who's never heard of it before..."

   SOCRATIC QUESTIONING:
   - Ask follow-up "why" questions to dig deeper
   - "What makes you think that?"
   - "Could there be another explanation?"

   TRUE/FALSE OR WOULD YOU RATHER:
   - "Would you say it's true that...?"
   - "Would you rather [option A] or [option B]? Why?"

3. Response Style:
   - Short, conversational questions (1-2 sentences max)
   - One concept at a time
   - Use their exact words
   - Add human touches (e.g., "That's interesting!")
   - NO long explanations
   - NO multiple questions at once
   - DO correct a statement if it is false, explain why if it's false
   - NEVER use "AI:" prefix in your responses

REMEMBER: Keep it casual and focused on ONE thing. Make them explain. Build up gradually. Occasionally use metaphors and varied question types to make the conversation more engaging."""),
        ("human", """Last response: {input}

Choose ONE aspect to explore deeper. Respond with a single, focused question that:
1. Builds directly on their words
2. Matches their current understanding level
3. Helps them explain the concept better
4. Feels natural and conversational

Occasionally (about 1/3 of the time), use one of these techniques to make your question more engaging:
- Frame your question using a metaphor or analogy
- Ask them to rank or compare concepts
- Present a hypothetical scenario
- Use Socratic questioning to dig deeper
- Offer a true/false statement or "would you rather" choice

Remember to keep it conversational and focused on ONE thing at a time.""")
    ])
    
    # Prepare related conceptskk
    # Create chain
    chain = prompt | llm_model
    
    # Create streaming handler
    handler = StreamingHandler()
    
    # Generate response
    response = chain.invoke({
        "input": user_message,
        "context": context,
        "history": formatted_history,
        "depth": teaching_state["depth_level"],
    }, config={"callbacks": [handler]})
    
    # Add to conversation history
    conversation_history.append(response)
    
    return jsonify({
        "response": response,
        "tokens": handler.tokens  # For streaming implementation on frontend
    })

@app.route('/test', methods=['GET'])
def test_endpoint():
    """Simple test endpoint to verify server functionality and CORS"""
    print("Test endpoint called!") 
    return jsonify({"status": "ok", "message": "Server is running and CORS is configured correctly"})

@app.after_request
def add_cors_headers(response):
    """Add CORS headers to all responses"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flask server for Enso Electron app')
    parser.add_argument('--port', type=int, default=8080, help='Port to run the server on')
    args = parser.parse_args()
    
    print(f"Starting Flask server on port {args.port}...")
    print("CORS settings:")
    print(f"- Origins: {app.config.get('CORS_ORIGINS', ['*'])}")
    print(f"- Methods: {app.config.get('CORS_METHODS', ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])}")
    print(f"- Headers: {app.config.get('CORS_HEADERS', ['Content-Type', 'Authorization', 'X-Requested-With'])}")
    
    # Force stdout to flush after each print for clearer logging
    import sys
    sys.stdout.flush()
    
    # Capture standard output and stderr for better debugging
    import logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('werkzeug')
    logger.setLevel(logging.INFO)
    
    # Use threaded=True for better handling of concurrent requests
    app.run(debug=True, port=args.port, threaded=True, use_reloader=False)