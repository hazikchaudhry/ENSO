from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                           QTextEdit, QLineEdit, QFrame, QComboBox, QProgressBar, 
                           QMessageBox, QScrollArea, QSizePolicy)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QIcon, QPixmap, QColor, QLinearGradient, QPalette, QFont, QTextCursor
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
import os
import sys
import fitz  # PyMuPDF for PDF processing
from docx import Document  # python-docx for DOCX processing
import time
import random

class ProcessingThread(QThread):
    finished = pyqtSignal(bool, str)
    progress = pyqtSignal(str)

    def __init__(self, file_path, start_page, end_page, model_name):
        super().__init__()
        self.file_path = file_path
        self.start_page = start_page
        self.end_page = end_page
        self.model_name = model_name
        self.vectorstore = None
        self.llm = None

    def run(self):
        try:
            self.progress.emit("Extracting text from document...")
            # Extract text based on file type
            if self.file_path.endswith('.pdf'):
                text = self.process_pdf()
            elif self.file_path.endswith('.docx'):
                text = self.process_docx()
            elif self.file_path.endswith('.txt'):
                text = self.process_txt()
            else:
                self.finished.emit(False, "Unsupported file format")
                return

            self.progress.emit("Creating text chunks...")
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_text(text)

            self.progress.emit("Creating embeddings...")
            # Create embeddings and vectorstore
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={'device': 'cpu'})
            self.vectorstore = FAISS.from_texts(chunks, embeddings)

            self.progress.emit("Initializing AI model...")
            # Initialize LLM
            self.llm = OllamaLLM(model=self.model_name)

            self.finished.emit(True, "Processing complete!")
        except Exception as e:
            self.finished.emit(False, f"Error: {str(e)}")

    def process_pdf(self):
        """Extract text from PDF within specified page range"""
        text = ""
        with fitz.open(self.file_path) as doc:
            start = self.start_page - 1 if self.start_page else 0
            end = min(self.end_page if self.end_page else doc.page_count, doc.page_count)
            for page_num in range(start, end):
                text += doc[page_num].get_text()
        return text

    def process_docx(self):
        """Extract text from DOCX"""
        doc = Document(self.file_path)
        return "\n".join([para.text for para in doc.paragraphs])

    def process_txt(self):
        """Extract text from TXT"""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            return f.read()

class ChatWorker(QThread):
    token_received = pyqtSignal(str)  # Emit tokens as they are generated
    finished = pyqtSignal(str)        # Emit the final response

    def __init__(self, question: str, context: str, conversation_history: list, teaching_state: dict):
        super().__init__()
        self.question = question
        self.context = context
        self.conversation_history = conversation_history
        self.teaching_state = teaching_state

    def run(self):
        try:
            # Format conversation history for better context
            formatted_history = ""
            if len(self.conversation_history) >= 2:  # Need at least one exchange
                # Get the last 6 messages (3 exchanges)
                recent_history = self.conversation_history[-6:]
                
                # Format as alternating User/AI messages
                for i in range(0, len(recent_history), 2):
                    if i < len(recent_history):
                        formatted_history += f"User: {recent_history[i]}\n"
                    if i+1 < len(recent_history):
                        formatted_history += f"AI: {recent_history[i+1]}\n\n"
            
            # Extract key concepts from user's response
            user_concepts = self.question.lower()
            
            # Find relevant context from vectorstore
            relevant_context = self.context if self.context else ""
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a friendly, curious learning companion using the Feynman Technique. Your goal is to have a natural, conversational exchange that guides the user to deeper understanding through thoughtful questions.

CURRENT STATE:
Topic: {topic}
Understanding: Level {understanding} (0: unclear → 3: excellent)
Depth: Level {depth} (0: basics → 2: advanced)

CONVERSATION HISTORY:
{history}

CONTEXT (Use to inform questions, never teach directly):
{context}

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
   - NO teaching or correcting
   - NEVER use "AI:" prefix in your responses

EXAMPLE FLOW:
User: "Neural networks process data"
You: "Oh, interesting! 💡 I'm curious - what actually happens when data first enters a neural network?"

User: "Well, the input layer receives the data"
You: "Got it! And what form does that data take when it reaches the input layer?"

User: "It's usually numerical values"
You: "I see! Would you say it's like translating a language - turning real-world information into numbers the network can understand?"

Remember: Keep it casual and focused on ONE thing. Make them explain. Build up gradually. Occasionally use metaphors and varied question types to make the conversation more engaging."""),
                ("human", """Last response: {input}

Choose ONE aspect to explore deeper. Respond with a single, focused question that:
1. Builds directly on their words
2. Matches their current understanding level
3. Helps them explain the concept better
4. Feels natural and conversational

Occasionally (about 30% of the time), use one of these techniques to make your question more engaging:
- Frame your question using a metaphor or analogy
- Ask them to rank or compare concepts
- Present a hypothetical scenario
- Use Socratic questioning to dig deeper
- Offer a true/false statement or "would you rather" choice

Remember to keep it conversational and focused on ONE thing at a time.""")
            ])
            
            chain = prompt | llm_model
            full_response = ""
            
            # Start with newline
            self.token_received.emit("\n")
            
            for chunk in chain.stream({
                "input": self.question,
                "context": relevant_context,
                "history": formatted_history,
                "topic": self.teaching_state["current_topic"] or "not set",
                "understanding": self.teaching_state["understanding_level"],
                "depth": self.teaching_state["depth_level"]
            }):
                full_response += chunk
                self.token_received.emit(chunk)
                
            self.finished.emit(full_response)
        except Exception as e:
            self.token_received.emit(f"\nError: {str(e)}")
            self.finished.emit("")

class FeynmanChatbot(QMainWindow):
    """Main application window for the Feynman Chatbot"""
    
    def __init__(self):
        """Initialize the Feynman Chatbot application"""
        super().__init__()
        
        # Initialize attributes
        self.vectorstore = None
        self.document_text = ""
        self.conversation_history = []
        self.conversation_state = "introduction"
        self.last_user_message = ""
        self.last_ai_message = ""
        self.last_user_message_pos = 0
        self.last_ai_message_pos = 0
        self.current_ai_response = None
        self.last_concept = None
        
        # Set up UI
        self.init_ui()
        self.chat_worker = None
        self.document_processed = False
        
        # Initialize Ollama model with streaming
        global llm_model
        llm_model = OllamaLLM(
            model="mistral",
            temperature=0.7,
            streaming=True
        )
        
    def init_ui(self):
        self.setWindowTitle("Enso")
        self.setGeometry(100, 100, 900, 700)
        
        # Set up the main background gradient
        main_palette = self.palette()
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor(245, 245, 250))
        gradient.setColorAt(1, QColor(235, 235, 245))
        main_palette.setBrush(QPalette.ColorRole.Window, gradient)
        self.setPalette(main_palette)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # Header with title
        header_layout = QHBoxLayout()
        title_label = QLabel("Enso")
        title_label.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        main_layout.addLayout(header_layout)
        
        # Document upload tile
        self.doc_tile = QFrame()
        self.doc_tile.setObjectName("docTile")
        self.doc_tile.setStyleSheet("""
            #docTile {
                background-color: black;
                border-radius: 15px;
                border: 1px solid #e0e0e0;
            }
        """)
        doc_tile_layout = QVBoxLayout(self.doc_tile)
        
        # Tile header
        tile_header = QLabel("Upload Document")
        tile_header.setFont(QFont("Arial", 16, ))
        tile_header.setStyleSheet("color: white;")
        doc_tile_layout.addWidget(tile_header)
        
        # Document selection controls
        doc_controls_layout = QHBoxLayout()
        
        # Upload button with icon
        self.upload_btn = QPushButton()
        self.upload_btn.setIcon(QIcon("/Users/hazikchaudhry/Documents/new feyman/icons/Add_square_duotone.png"))
        self.upload_btn.setIconSize(QSize(24, 24))
        self.upload_btn.setText("Upload Document")
        self.upload_btn.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f5;
                border-radius: 8px;
                padding: 8px 15px;
                font-size: 14px;
                border: none;
            }
            QPushButton:hover {
                background-color: #e0e0e5;
            }
        """)
        self.upload_btn.clicked.connect(self.upload_document)
        doc_controls_layout.addWidget(self.upload_btn)
        
        # Page range inputs
        page_range_layout = QHBoxLayout()
        page_range_layout.setSpacing(5)
        
        self.page_start = QLineEdit()
        self.page_start.setPlaceholderText("Start Page")
        self.page_start.setFixedWidth(100)
        self.page_start.setStyleSheet("""
            QLineEdit {
                border-radius: 8px;
                padding: 8px;
                background-color: #f5f5f5;
                border: 1px solid #e0e0e0;
            }
        """)
        page_range_layout.addWidget(self.page_start)
        
        page_range_layout.addWidget(QLabel("-"))
        
        self.page_end = QLineEdit()
        self.page_end.setPlaceholderText("End Page")
        self.page_end.setFixedWidth(100)
        self.page_end.setStyleSheet("""
            QLineEdit {
                border-radius: 8px;
                padding: 8px;
                background-color: #f5f5f5;
                border: 1px solid #e0e0e0;
            }
        """)
        page_range_layout.addWidget(self.page_end)
        
        doc_controls_layout.addLayout(page_range_layout)
        
        # Model selector
        model_layout = QHBoxLayout()
        model_label = QLabel("Model:")
        model_layout.addWidget(model_label)
        
        self.model_selector = QComboBox()
        self.model_selector.addItems(["deepseek-r1:7b", "mistral", "llama2"])
        self.model_selector.setStyleSheet("""
            QComboBox {
                border-radius: 8px;
                padding: 8px;
                background-color: #f5f5f5;
                border: 1px solid #e0e0e0;
                min-width: 150px;
            }
        """)
        model_layout.addWidget(self.model_selector)
        
        # Settings button
        self.settings_btn = QPushButton()
        self.settings_btn.setIcon(QIcon("/Users/hazikchaudhry/Documents/new feyman/icons/Setting_line.png"))
        self.settings_btn.setIconSize(QSize(20, 20))
        self.settings_btn.setFixedSize(40, 40)
        self.settings_btn.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f5;
                border-radius: 20px;
                border: none;
            }
            QPushButton:hover {
                background-color: #e0e0e5;
            }
        """)
        model_layout.addWidget(self.settings_btn)
        
        doc_controls_layout.addLayout(model_layout)
        doc_tile_layout.addLayout(doc_controls_layout)
        
        # Progress indicators
        progress_layout = QHBoxLayout()
        self.status_label = QLabel()
        self.status_label.setStyleSheet("color: #555;")
        progress_layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border-radius: 5px;
                background-color: #f0f0f0;
                text-align: center;
                height: 10px;
            }
            QProgressBar::chunk {
                background-color: #4a86e8;
                border-radius: 5px;
            }
        """)
        progress_layout.addWidget(self.progress_bar)
        
        doc_tile_layout.addLayout(progress_layout)
        main_layout.addWidget(self.doc_tile)
        
        # Chat area
        chat_container = QFrame()
        chat_container.setObjectName("chatContainer")
        chat_container.setStyleSheet("""
            #chatContainer {
                background-color: white;
                border-radius: 15px;
                border: 1px solid #e0e0e0;
            }
        """)
        chat_layout = QVBoxLayout(chat_container)
        
        # Custom QTextEdit for styled messages
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("""
            QTextEdit {
                border: none;
                background-color: white;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 14px;
                line-height: 1.5;
            }
            QScrollBar:vertical {
                border: none;
                background: #f1f1f1;
                width: 8px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #c1c1c1;
                min-height: 30px;
                border-radius: 4px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
                height: 0px;
            }
        """)
        
        # Set a fixed policy to ensure the chat display expands properly
        self.chat_display.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Set document margins for better text display
        self.chat_display.document().setDocumentMargin(10)
        
        chat_layout.addWidget(self.chat_display)
        
        # Input area
        input_frame = QFrame()
        input_frame.setStyleSheet("""
            QFrame {
                background-color: #f5f5f5;
                border-radius: 10px;
                border: 1px solid #e0e0e0;
            }
        """)
        input_layout = QVBoxLayout(input_frame)
        input_layout.setContentsMargins(10, 5, 10, 5)
        
        # Rich text editing toolbar
        toolbar_layout = QHBoxLayout()
        
        # Bold button
        bold_btn = QPushButton("B")
        bold_btn.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        bold_btn.setFixedSize(30, 30)
        bold_btn.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border-radius: 5px;
                border: 1px solid #d0d0d0;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        bold_btn.clicked.connect(self.apply_bold)
        toolbar_layout.addWidget(bold_btn)
        
        # Italic button
        italic_btn = QPushButton("I")
        italic_btn.setFont(QFont("Arial", 10, QFont.Weight.Normal, True))
        italic_btn.setFixedSize(30, 30)
        italic_btn.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border-radius: 5px;
                border: 1px solid #d0d0d0;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        italic_btn.clicked.connect(self.apply_italic)
        toolbar_layout.addWidget(italic_btn)
        
        # Bullet list button
        bullet_btn = QPushButton("•")
        bullet_btn.setFont(QFont("Arial", 12))
        bullet_btn.setFixedSize(30, 30)
        bullet_btn.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border-radius: 5px;
                border: 1px solid #d0d0d0;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        bullet_btn.clicked.connect(self.apply_bullet)
        toolbar_layout.addWidget(bullet_btn)
        
        # Add spacer to push buttons to the left
        toolbar_layout.addStretch()
        
        input_layout.addLayout(toolbar_layout)
        
        # Replace QLineEdit with QTextEdit for taller input with rich text capabilities
        self.user_input = QTextEdit()
        self.user_input.setPlaceholderText("What's in your mind?...")
        self.user_input.setStyleSheet("""
            QTextEdit {
                border: none;
                background-color: transparent;
                padding: 8px;
                font-size: 14px;
                color: black;
                min-height: 60px;
                max-height: 120px;
            }
        """)
        self.user_input.setAcceptRichText(True)
        self.user_input.setTabChangesFocus(True)
        input_layout.addWidget(self.user_input)
        
        # Add send button in a separate layout
        send_layout = QHBoxLayout()
        send_layout.addStretch()
        
        self.send_btn = QPushButton()
        self.send_btn.setIcon(QIcon("/Users/hazikchaudhry/Documents/new feyman/icons/Arrow_right.png"))
        self.send_btn.setIconSize(QSize(20, 20))
        self.send_btn.setFixedSize(40, 40)
        self.send_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a86e8;
                border-radius: 20px;
                border: none;
            }
            QPushButton:hover {
                background-color: #3a76d8;
            }
        """)
        self.send_btn.clicked.connect(self.process_user_response)
        send_layout.addWidget(self.send_btn)
        
        input_layout.addLayout(send_layout)
        
        chat_layout.addWidget(input_frame)
        
        main_layout.addWidget(chat_container, 1)
        
        # Initially disable chat until document is processed
        self.user_input.setEnabled(False)
        self.send_btn.setEnabled(False)
        
        # Initialize components
        self.processing_thread = None
        self.vectorstore = None
        self.chain = None
        self.last_ai_message_pos = None
        self.last_user_message_pos = None

    def closeEvent(self, event):
        """Clean up resources before closing"""
        if self.chat_worker and self.chat_worker.isRunning():
            self.chat_worker.quit()
            self.chat_worker.wait()
        event.accept()

    def upload_document(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Document",
            "",
            "Documents (*.pdf *.docx *.txt)"
        )
        
        if file_path:
            try:
                # Get page range
                start_page = int(self.page_start.text()) if self.page_start.text() else 1
                end_page = int(self.page_end.text()) if self.page_end.text() else None
                
                # Disable UI during processing
                self.upload_btn.setEnabled(False)
                self.user_input.setEnabled(False)
                self.send_btn.setEnabled(False)
                self.progress_bar.setVisible(True)
                self.progress_bar.setRange(0, 0)  # Indeterminate progress
                
                # Start processing thread
                self.processing_thread = ProcessingThread(
                    file_path, start_page, end_page, 
                    self.model_selector.currentText()
                )
                self.processing_thread.progress.connect(self.update_progress)
                self.processing_thread.finished.connect(self.processing_complete)
                self.processing_thread.start()
                
            except ValueError:
                QMessageBox.warning(self, "Error", "Please enter valid page numbers")

    def update_progress(self, message):
        self.status_label.setText(message)

    def processing_complete(self, success, message):
        self.progress_bar.setVisible(False)
        self.upload_btn.setEnabled(True)
        
        if success:
            self.vectorstore = self.processing_thread.vectorstore
            self.user_input.setEnabled(True)
            self.send_btn.setEnabled(True)
            self.document_processed = True
            
            # Collapse document tile
            self.doc_tile.setMaximumHeight(0)
            self.doc_tile.setVisible(False)
            
            # Display initial greeting
            self.append_ai_message("Hey there! 👋 I'm excited to explore this document with you! Think of me as your curious learning buddy - we're like detectives looking for interesting clues in the text. What caught your eye? Even a simple word or idea is perfect to start our investigation! 🔍")
        else:
            QMessageBox.critical(self, "Error", message)
        
        self.processing_thread = None

    def process_user_response(self):
        # Get text from QTextEdit instead of QLineEdit
        user_text = self.user_input.toPlainText().strip()
        if not user_text:
            return
        
        # Clear input box and disable until response is complete
        self.user_input.clear()
        self.user_input.setEnabled(False)
        
        # Display user message with proper styling
        self.append_user_message(user_text)
        
        # Add to conversation history as a user message
        self.conversation_history.append(user_text)
        
        # Ensure the chat display is scrolled to the bottom
        self.chat_display.verticalScrollBar().setValue(
            self.chat_display.verticalScrollBar().maximum()
        )
        
        try:
            # Update conversation state based on user response
            self._update_conversation_state(user_text)

            # Get relevant context from vectorstore
            context = ""
            if hasattr(self, 'vectorstore') and self.vectorstore:
                results = self.vectorstore.similarity_search(user_text, k=2)
                context = "\n".join([doc.page_content for doc in results])

            # Determine teaching state based on conversation state
            teaching_state = self._get_teaching_state()

            # Create and start chat worker
            self.chat_worker = ChatWorker(
                question=user_text,
                context=context,
                conversation_history=self.conversation_history,
                teaching_state={"current_topic": self.last_concept, "understanding_level": 1, "depth_level": 0}
            )
            
            # Connect signals
            self.chat_worker.token_received.connect(self._handle_token)
            self.chat_worker.finished.connect(self._handle_response_finished)
            
            # Start processing
            self.chat_worker.start()

        except Exception as e:
            self.append_ai_message("Hmm, I didn't quite catch that. Could you try saying that in a different way? 🤔")
            self.conversation_history.append("Hmm, I didn't quite catch that. Could you try saying that in a different way? 🤔")
            print(f"Error in process_user_response: {str(e)}")
            self.user_input.setEnabled(True)
            
    def append_user_message(self, text):
        """Append a styled user message to the chat display"""
        # Save the message for edit functionality
        self.last_user_message = text
        
        # Create HTML for user message with ChatGPT-like styling
        html = f"""
        <div style="width: 100%; background-color: #f7f7f8; padding: 12px 0; border-bottom: 1px solid #e5e5e5;">
            <div style="max-width: 90%; margin: 0 auto; color: #343541; font-size: 14px;">
                <div style="display: flex; align-items: flex-start;">
                    <div style="width: 30px; height: 30px; background-color: #5436DA; border-radius: 50%; 
                              display: flex; justify-content: center; align-items: center; margin-right: 15px; flex-shrink: 0;">
                        <span style="color: white; font-weight: bold;">U</span>
                    </div>
                    <div style="flex-grow: 1;">
                        {text}
                    </div>
                </div>
            </div>
        </div>
        """
        self.chat_display.insertHtml(html)
        
        # Ensure the view scrolls to show the new message
        self.chat_display.moveCursor(QTextCursor.MoveOperation.End)
        self.chat_display.ensureCursorVisible()
        self.last_user_message_pos = self.chat_display.textCursor().position()
        
    def append_ai_message(self, text):
        """Append a styled AI message to the chat display"""
        # Save the message for refresh functionality
        self.last_ai_message = text
        
        # Create HTML for AI message with ChatGPT-like styling
        html = f"""
        <div style="width: 100%; background-color: white; padding: 12px 0; border-bottom: 1px solid #e5e5e5;">
            <div style="max-width: 90%; margin: 0 auto; color: #343541; font-size: 14px;">
                <div style="display: flex; align-items: flex-start;">
                    <div style="width: 30px; height: 30px; background-color: #10a37f; border-radius: 50%; 
                              display: flex; justify-content: center; align-items: center; margin-right: 15px; flex-shrink: 0;">
                        <span style="color: white; font-weight: bold;">AI</span>
                    </div>
                    <div style="flex-grow: 1;">
                        {text}
                    </div>
                </div>
            </div>
        </div>
        """
        self.chat_display.insertHtml(html)
        
        # Ensure the view scrolls to show the new message
        self.chat_display.moveCursor(QTextCursor.MoveOperation.End)
        self.chat_display.ensureCursorVisible()
        self.last_ai_message_pos = self.chat_display.textCursor().position()
    
    def _handle_token(self, token: str):
        """Handle incoming tokens from the chat worker"""
        # If this is the first token, remove the typing indicator and start a new AI message
        if self.current_ai_response is None:
            self.current_ai_response = ""
            
            # Create a new AI message container with ChatGPT-like styling
            html = """
            <div style="width: 100%; background-color: white; padding: 12px 0; border-bottom: 1px solid #e5e5e5;">
                <div style="max-width: 90%; margin: 0 auto; color: #343541; font-size: 14px;">
                    <div style="display: flex; align-items: flex-start;">
                        <div style="width: 30px; height: 30px; background-color: #10a37f; border-radius: 50%; 
                                  display: flex; justify-content: center; align-items: center; margin-right: 15px; flex-shrink: 0;">
                            <span style="color: white; font-weight: bold;">AI</span>
                        </div>
                        <div id="streaming_response" style="flex-grow: 1;">
                        </div>
                    </div>
                </div>
            </div>
            """
            self.chat_display.insertHtml(html)
            
        self.current_ai_response += token
        
        # Find the streaming response div and update its content
        document = self.chat_display.document()
        cursor = QTextCursor(document)
        cursor.movePosition(QTextCursor.MoveOperation.End)
        
        # Find the streaming response div
        found = cursor.movePosition(QTextCursor.MoveOperation.PreviousBlock)
        if found:
            cursor.movePosition(QTextCursor.MoveOperation.EndOfBlock)
            cursor.insertText(token)
            
        # Keep the scroll at the bottom
        self.chat_display.verticalScrollBar().setValue(
            self.chat_display.verticalScrollBar().maximum()
        )
        
    def _handle_response_finished(self, full_response: str):
        """Handle when the chat response is complete"""
        # Clear the current response buffer
        temp_response = self.current_ai_response
        self.current_ai_response = None
        
        # Remove the streaming response div
        document = self.chat_display.document()
        cursor = QTextCursor(document)
        cursor.movePosition(QTextCursor.MoveOperation.End)
        
        # Find and remove the streaming message
        found = cursor.movePosition(QTextCursor.MoveOperation.PreviousBlock)
        if found:
            # Move to the start of the entire div container
            for _ in range(8):  # Navigate up through the HTML structure (adjusted for new format)
                if not cursor.movePosition(QTextCursor.MoveOperation.PreviousBlock):
                    break
            
            # Select from current position to the end of the streaming message
            cursor.movePosition(QTextCursor.MoveOperation.NextBlock, cursor.MoveMode.KeepAnchor, 8)
            cursor.removeSelectedText()
        
        # Add the complete AI message with proper styling
        self.append_ai_message(full_response)
        
        # Add to conversation history
        self.conversation_history.append(full_response)
        
        # Ensure the chat display is scrolled to the bottom
        self.chat_display.verticalScrollBar().setValue(
            self.chat_display.verticalScrollBar().maximum()
        )
        
        # Re-enable input box
        self.user_input.setEnabled(True)
        
        # Set focus back to the input box for immediate typing
        self.user_input.setFocus()
    
    def _update_conversation_state(self, user_text):
        """Update conversation state based on user's response"""
        # Simple state tracking for conversation
        if self.conversation_state == "introduction":
            self.conversation_state = "exploring"
            
        # Extract potential topics from user text
        text_lower = user_text.lower()
        key_concepts = ["neural network", "transformer", "attention", "encoder", "decoder", "layer"]
        for concept in key_concepts:
            if concept in text_lower:
                # Just update the current topic, no complex state object
                self.last_concept = concept
                break

    def _get_teaching_state(self):
        """Determine teaching state based on conversation state"""
        # Simple teaching state mapping
        if self.conversation_state == "introduction":
            return "introduction"
        elif self.conversation_state == "exploring":
            return "exploring"
        else:
            return "challenge_understanding"

    # Add text formatting methods
    def apply_bold(self):
        """Apply bold formatting to selected text"""
        cursor = self.user_input.textCursor()
        if cursor.hasSelection():
            selected_text = cursor.selectedText()
            cursor.removeSelectedText()
            cursor.insertHtml(f"<b>{selected_text}</b>")
        else:
            # If no selection, insert bold tags and position cursor between them
            cursor.insertHtml("<b></b>")
            # Move cursor back by 4 characters (</b> length)
            for _ in range(4):
                cursor.movePosition(QTextCursor.MoveOperation.Left)
            self.user_input.setTextCursor(cursor)
    
    def apply_italic(self):
        """Apply italic formatting to selected text"""
        cursor = self.user_input.textCursor()
        if cursor.hasSelection():
            selected_text = cursor.selectedText()
            cursor.removeSelectedText()
            cursor.insertHtml(f"<i>{selected_text}</i>")
        else:
            # If no selection, insert italic tags and position cursor between them
            cursor.insertHtml("<i></i>")
            # Move cursor back by 4 characters (</i> length)
            for _ in range(4):
                cursor.movePosition(QTextCursor.MoveOperation.Left)
            self.user_input.setTextCursor(cursor)
    
    def apply_bullet(self):
        """Insert a bullet point at the current cursor position"""
        cursor = self.user_input.textCursor()
        # Move to start of line
        cursor.movePosition(QTextCursor.MoveOperation.StartOfLine)
        # Insert bullet point
        cursor.insertHtml("• ")
        self.user_input.setTextCursor(cursor)

def main():
    app = QApplication(sys.argv)
    window = FeynmanChatbot()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()