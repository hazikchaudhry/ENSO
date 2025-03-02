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

    def _analyze_understanding(self, user_text: str, context: str) -> float:
        """Let the AI assess understanding through natural conversation"""
        # The understanding level will be determined by the AI model's assessment
        # through the natural flow of conversation, considering:
        # - Depth of explanations
        # - Concept relationships
        # - Contextual relevance
        # - Response coherence
        
        # For now, maintain a moderate understanding level to allow
        # the conversation to flow naturally
        return 0.5

    def _update_understanding_level(self, user_text: str, context: str):
        """Update the understanding level based on user's response quality"""
        current_level = self.teaching_state["understanding_level"]
        understanding_score = self._analyze_understanding(user_text, context)
        
        # Gradual level adjustment
        if understanding_score >= 0.8 and current_level < 3:
            self.teaching_state["understanding_level"] = min(3, current_level + 1)
        elif understanding_score <= 0.3 and current_level > 0:
            self.teaching_state["understanding_level"] = max(0, current_level - 1)
        elif 0.3 < understanding_score < 0.8:
            # Maintain current level but track progress
            self.teaching_state["progress_to_next"] = understanding_score

    def _detect_misconceptions(self, user_text: str, context: str) -> list:
        """Detect potential misconceptions in user's response by comparing with context"""
        misconceptions = []
        
        # Convert to lowercase for case-insensitive comparison
        user_text = user_text.lower()
        context = context.lower()
        
        # Common misconception patterns
        contradictions = [
            ("always", "sometimes"),
            ("never", "can"),
            ("only", "also"),
            ("all", "some")
        ]
        
        # Check for direct contradictions
        for word1, word2 in contradictions:
            if word1 in user_text and word2 in context:
                misconceptions.append(f"Potential oversimplification with '{word1}'")
        
        # Check for incorrect relationships
        if "because" in user_text:
            cause_effect = user_text.split("because")[1].strip()
            if cause_effect and cause_effect not in context:
                misconceptions.append("Potential incorrect cause-effect relationship")
        
        # Check for absolute statements
        absolute_words = ["always", "never", "all", "none", "every", "only"]
        for word in absolute_words:
            if word in user_text and word not in context:
                misconceptions.append(f"Potential overgeneralization with '{word}'")
        
        return misconceptions

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
            
            # Update understanding level based on user's response
            self._update_understanding_level(user_concepts, relevant_context)
            
            # Adjust question complexity based on teaching state and user understanding
            state = self.teaching_state
            question_style = "basic"
            
            # Calculate complexity level based on conversation length and understanding
            complexity_level = min(3, len(self.conversation_history) // 4 + state["understanding_level"])
            
            # Only check for misconceptions if this isn't the first interaction
            misconception_detected = False
            hint_prompt = ""
            
            if len(self.conversation_history) > 2:  # At least one full exchange has occurred
                if state["understanding_level"] > 0:
                    # Simple keyword-based misconception detection
                    misconceptions = self._detect_misconceptions(user_concepts, relevant_context)
                    if misconceptions:
                        question_style = "socratic"
                        misconception_detected = True
                        hint_prompt = "\nGently guide the user to discover any potential misunderstandings through questions."
            
            # Choose question style based on conversation progress and complexity
            if not misconception_detected:
                if complexity_level == 0:
                    question_style = "basic"  # Start with foundational questions
                elif complexity_level == 1:
                    styles = ["basic", "comparison", "metaphor"]
                    question_style = random.choice(styles)
                elif complexity_level == 2:
                    styles = ["comparison", "scenario", "metaphor", "socratic"]
                    question_style = random.choice(styles)
                else:
                    styles = ["socratic", "scenario", "metaphor"]
                    question_style = random.choice(styles)
            
            # Include related concepts for deeper exploration
            related_concepts_prompt = ""
            if state.get("related_concepts"):
                related = ", ".join(state["related_concepts"][:3])
                connection_prompt = "\nExplore connections between these concepts and identify patterns or relationships. "
                depth_prompt = "\nConsider how these concepts build upon or influence each other. "
                related_concepts_prompt = f"\nRelated concepts to explore: {related}. {connection_prompt if complexity_level >= 2 else ''}{depth_prompt if complexity_level >= 3 else ''}"
            
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

Occasionally (about 1/3 of the time), use one of these techniques to make your question more engaging:
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
                "depth": self.teaching_state["depth_level"],
                "question_style": question_style,
                "related_concepts_prompt": related_concepts_prompt
            }):
                full_response += chunk
                self.token_received.emit(chunk)
                
            self.finished.emit(full_response)
        except Exception as e:
            self.token_received.emit(f"\nError: {str(e)}")
            self.finished.emit("")

class Enso(QMainWindow):
    """Main application window for the Enso application"""
    
    def __init__(self):
        """Initialize the Enso application"""
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
            model="mistral:7b-instruct",
            temperature=0.7,
            streaming=True
        )
        
    def init_ui(self):
        self.setWindowTitle("Enso")
        self.setGeometry(100, 100, 900, 700)
        
        # Set up the main background gradient
        main_palette = self.palette()
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor('#588157'))
        gradient.setColorAt(0.5, QColor('#3A5A40'))
        gradient.setColorAt(1, QColor('#344E41'))
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
        title_label.setStyleSheet("color: #5006D77;")
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        main_layout.addLayout(header_layout)
        
        # Document upload tile
        self.doc_tile = QFrame()
        self.doc_tile.setObjectName("docTile")
        self.doc_tile.setStyleSheet("""
            #docTile {
                background: #344E41;
                border-radius: 15px;
                border: none;
            }
        """)
        doc_tile_layout = QVBoxLayout(self.doc_tile)
        
        # Tile header
        tile_header = QLabel("Upload Document")
        tile_header.setFont(QFont("Arial", 16, ))
        tile_header.setStyleSheet("color: white;")
        doc_tile_layout.addWidget(tile_header)
        
        # Add informative text
        info_layout = QHBoxLayout()
        
        # Requirements section
        requirements_text = QLabel(
            "Requirements:\n" 
            "• Ollama must be running locally\n"
            "• Supports PDF, DOCX, and TXT files"
        )
        requirements_text.setWordWrap(True)
        requirements_text.setStyleSheet("color: #DAD7CD; font-size: 17px; margin: 3px 0;")
        info_layout.addWidget(requirements_text)
        
        # How it works section
        workflow_text = QLabel(
            "How it works:\n"
            "1. Select document & page range\n"
            "2. Choose AI model\n"
            "3. Process text & create embeddings\n"
            "4. Start interactive learning"
        )
        workflow_text.setWordWrap(True)
        workflow_text.setStyleSheet("color: #DAD7CD; font-size: 15px; margin: 4px 0;")
        info_layout.addWidget(workflow_text)
        
        doc_tile_layout.addLayout(info_layout)
        
        # Document selection controls
        doc_controls_layout = QHBoxLayout()
        
        # Initialize upload button
        self.upload_btn = QPushButton("Upload")
        self.upload_btn.setStyleSheet("""
            QPushButton {
                background-color: #DAD7CD;
                border-radius: 8px;
                padding: 8px;
                border: 1px solid #344E41;
                color: #588157;
            }
            QPushButton:hover {
                background-color: #E9EDC9;
            }
        """)
        
        # Status label for progress updates
        self.status_label = QLabel()
        self.status_label.setStyleSheet("color: #DAD7CD;")
        doc_tile_layout.addWidget(self.status_label)
        self.upload_btn.clicked.connect(self.upload_document)
        
        # Initialize model selector
        self.model_selector = QComboBox()
        self.model_selector.addItems(["mistral:7b-instruct"])
        doc_controls_layout.addWidget(self.model_selector)
        
        
        # Initialize progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        
        # Initialize page range inputs
        self.page_start = QLineEdit()
        self.page_start.setPlaceholderText("Start Page")
        self.page_end = QLineEdit()
        self.page_end.setPlaceholderText("End Page")
        
        # Add page inputs to layout
        doc_controls_layout.addWidget(self.page_start)
        doc_controls_layout.addWidget(self.page_end)
        
        # Add the upload button to the layout
        doc_controls_layout.addWidget(self.upload_btn)
        
        # Add the progress bar to the layout
        doc_tile_layout.addWidget(self.progress_bar)
        
        # Add the document controls layout to the tile layout
        doc_tile_layout.addLayout(doc_controls_layout)
        
        # Add the document tile to the main layout
        main_layout.addWidget(self.doc_tile)
        
        # Create chat container
        chat_container = QFrame()
        chat_container.setObjectName("chatContainer")
        chat_layout = QVBoxLayout(chat_container)
        
        # Create scroll area for chat
        chat_scroll_area = QScrollArea()
        chat_scroll_area.setWidgetResizable(True)
        chat_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # Create content widget for scroll area
        chat_content_widget = QWidget()
        self.chat_content_layout = QVBoxLayout(chat_content_widget)
        
        # Initialize chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        
        # Status label for progress updates
        self.status_label = QLabel()
        self.status_label.setStyleSheet("color: #DAD7CD;")
        doc_tile_layout.addWidget(self.status_label)
        
        # Page range inputs styling
        page_input_style = """
            QLineEdit {
                border-radius: 8px;
                padding: 8px;
                background-color: #DAD7CD;
                border: 1px solid #344E41;
                color: #588157;
            }
        """
        self.page_start.setStyleSheet(page_input_style)
        self.page_end.setStyleSheet(page_input_style)
        
        # Model selector styling
        self.model_selector.setStyleSheet("""
            QComboBox {
                border-radius: 8px;
                padding: 8px;
                background-color: #DAD7CD;
                border: 1px solid #344E41;
                color: #588157;
                min-width: 150px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                border: none;
                background: #D4A373;
            }
        """)
        
        
        # Progress bar styling
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border-radius: 5px;
                background-color: #FAEDCD;
                text-align: center;
                height: 10px;
            }
            QProgressBar::chunk {
                background-color: #D4A373;
                border-radius: 5px;
            }
        """)
        
        # Chat container styling
        chat_container.setStyleSheet("""
            #chatContainer {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #E9EDC9,
                    stop:1 #DAD7CD);
                border: 1px solid #344E41;
                border-radius: 15px;
            }
        """)
        
        # Chat display styling
        self.chat_display.setStyleSheet("""
            QTextEdit {
                border: none;
                background-color: #FEFAE0;
                font-family: monospace;
                font-size: 14px;
                line-height: 1.5;
                padding: 10px;
                color: #0D1B2A;
            }
            QScrollBar:vertical {
                background: #E9EDC9;
                width: 12px;
                margin: 0px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #CCD5AE;
                min-height: 20px;
                border-radius: 6px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        self.chat_display.document().setDocumentMargin(10)
        
        # Add the chat display to the content layout
        self.chat_content_layout.addWidget(self.chat_display)
        
        # Set the content widget as the scroll area's widget
        chat_scroll_area.setWidget(chat_content_widget)
        
        # Add the scroll area to the chat layout
        chat_layout.addWidget(chat_scroll_area, 1)  # Give it a stretch factor of 1
        
        # Input area
        input_frame = QFrame()
        input_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #FEFAE0,
                    stop:1 #FAEDCD);
                border-radius: 10px;
                border: 1px solid #CCD5AE;
                margin: 10px;
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
                background-color: #FEFAE0;
                border-radius: 5px;
                border: 1px solid #CCD5AE;
                color: #D4A373;
            }
            QPushButton:hover {
                background-color: #E9EDC9;
            }
        """)
        
        # Italic button
        italic_btn = QPushButton("I")
        italic_btn.setFont(QFont("Arial", 10, QFont.Weight.Normal, True))
        italic_btn.setFixedSize(30, 30)
        italic_btn.setStyleSheet("""
            QPushButton {
                background-color: #FEFAE0;
                border-radius: 5px;
                border: 1px solid #CCD5AE;
                color: #D4A373;
            }
            QPushButton:hover {
                background-color: #E9EDC9;
            }
        """)
        
        # Bullet list button
        bullet_btn = QPushButton("•")
        bullet_btn.setFont(QFont("Arial", 12))
        bullet_btn.setFixedSize(30, 30)
        bullet_btn.setStyleSheet("""
            QPushButton {
                background-color: #FEFAE0;
                border-radius: 5px;
                border: 1px solid #CCD5AE;
                color: #D4A373;
            }
            QPushButton:hover {
                background-color: #E9EDC9;
            }
        """)
        
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
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #588157,
                    stop:1 #3A5A40);
                border-radius: 20px;
                border: none;
            }
            QPushButton:hover {
                background-color: #344E41;
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
        """Append a user message to the chat display"""
        # Save the message for edit functionality
        self.last_user_message = text
        
        # Add to conversation history
        self.conversation_history.append(text)
        
        # Move cursor to end and ensure proper spacing
        self.chat_display.moveCursor(QTextCursor.MoveOperation.End)
        cursor = self.chat_display.textCursor()
        
        # Add spacing if not at the start of the document
        if cursor.position() > 0:
            self.chat_display.insertPlainText("\n")
        
        # Insert user message with consistent formatting
        message = f"User: {text}\n"
        self.chat_display.insertPlainText(message)
        
        # Store position and ensure visibility
        self.last_user_message_pos = self.chat_display.textCursor().position()
        self.chat_display.ensureCursorVisible()

    def append_ai_message(self, text):
        """Append an AI message to the chat display"""
        # Save the message for refresh functionality
        self.last_ai_message = text
        
        # Only add to conversation history and display if it's not a streaming response
        if self.current_ai_response is None:
            # Add to conversation history if not already added
            if text not in self.conversation_history:
                self.conversation_history.append(text)
            
            # Move cursor to end and ensure proper spacing
            self.chat_display.moveCursor(QTextCursor.MoveOperation.End)
            cursor = self.chat_display.textCursor()
            
            # Add spacing if not at the start of the document
            if cursor.position() > 0:
                self.chat_display.insertPlainText("\n")
            
            # Insert AI message with consistent formatting
            message = f"ENSO: {text}\n"
            self.chat_display.insertPlainText(message)
            
            # Store position and ensure visibility
            self.last_ai_message_pos = self.chat_display.textCursor().position()
            self.chat_display.ensureCursorVisible()
    
    def _handle_token(self, token: str):
        """Handle incoming tokens from the chat worker"""
        # If this is the first token, start a new AI message
        if self.current_ai_response is None:
            self.current_ai_response = ""
            self.chat_display.moveCursor(QTextCursor.MoveOperation.End)
            cursor = self.chat_display.textCursor()
            
            # Add spacing if not at the start of the document
            if cursor.position() > 0:
                self.chat_display.insertPlainText("\n")
            
            self.chat_display.insertPlainText("ENSO: ")
        
        self.current_ai_response += token
        self.chat_display.insertPlainText(token)
        
        # Keep the scroll at the bottom
        self.chat_display.verticalScrollBar().setValue(
            self.chat_display.verticalScrollBar().maximum()
        )
        self.last_ai_message_pos = self.chat_display.textCursor().position()

    def _analyze_understanding(self, user_text: str, context: str) -> float:
        """Analyze user's understanding level based on response quality"""
        score = 0.0
        
        # Check for key concept usage
        context_keywords = set(context.lower().split())
        user_keywords = set(user_text.lower().split())
        concept_overlap = len(context_keywords.intersection(user_keywords)) / max(1, len(context_keywords))
        score += concept_overlap * 0.5  # Max 0.5 points for concept usage
        
        # Check for explanation quality
        if len(user_text.split()) >= 15:  # Reasonable explanation length
            score += 0.3
        if "because" in user_text.lower() or "therefore" in user_text.lower():  # Causal reasoning
            score += 0.4
        if "for example" in user_text.lower() or "like" in user_text.lower():  # Uses examples
            score += 0.4
        
        # Penalize for misconceptions
        misconceptions = self._detect_misconceptions(user_text, context)
        score = max(0.0, score - (len(misconceptions) * 0.3))
        
        return min(1.0, score)  # Normalize to 0-1 range

    def _update_understanding_level(self, user_text: str, context: str):
        """Update the understanding level based on user's response quality"""
        current_level = self.teaching_state["understanding_level"]
        understanding_score = self._analyze_understanding(user_text, context)
        
        # Gradual level adjustment
        if understanding_score >= 0.8 and current_level < 3:
            self.teaching_state["understanding_level"] = min(3, current_level + 1)
        elif understanding_score <= 0.3 and current_level > 0:
            self.teaching_state["understanding_level"] = max(0, current_level - 1)
        elif 0.3 < understanding_score < 0.8:
            # Maintain current level but track progress
            self.teaching_state["progress_to_next"] = understanding_score

    def _analyze_understanding(self, user_text: str, context: str) -> float:
        """Let the AI assess understanding through natural conversation"""
        # The understanding level will be determined by the AI model's assessment
        # through the natural flow of conversation, considering:
        # - Depth of explanations
        # - Concept relationships
        # - Contextual relevance
        # - Response coherence
        
        # For now, maintain a moderate understanding level to allow
        # the conversation to flow naturally
        return 0.5

    def _update_understanding_level(self, user_text: str, context: str):
        """Update the understanding level based on user's response quality"""
        current_level = self.teaching_state["understanding_level"]
        understanding_score = self._analyze_understanding(user_text, context)
        
        # Gradual level adjustment
        if understanding_score >= 0.8 and current_level < 3:
            self.teaching_state["understanding_level"] = min(3, current_level + 1)
        elif understanding_score <= 0.3 and current_level > 0:
            self.teaching_state["understanding_level"] = max(0, current_level - 1)
        elif 0.3 < understanding_score < 0.8:
            # Maintain current level but track progress
            self.teaching_state["progress_to_next"] = understanding_score

    def _detect_misconceptions(self, user_text: str, context: str) -> list:
        """Detect potential misconceptions in user's response by comparing with context"""
        misconceptions = []
        
        # Convert to lowercase for case-insensitive comparison
        user_text = user_text.lower()
        context = context.lower()
        
        # Common misconception patterns
        contradictions = [
            ("always", "sometimes"),
            ("never", "can"),
            ("only", "also"),
            ("all", "some")
        ]
        
        # Check for direct contradictions
        for word1, word2 in contradictions:
            if word1 in user_text and word2 in context:
                misconceptions.append(f"Potential oversimplification with '{word1}'")
        
        # Check for incorrect relationships
        if "because" in user_text:
            cause_effect = user_text.split("because")[1].strip()
            if cause_effect and cause_effect not in context:
                misconceptions.append("Potential incorrect cause-effect relationship")
        
        # Check for absolute statements
        absolute_words = ["always", "never", "all", "none", "every", "only"]
        for word in absolute_words:
            if word in user_text and word not in context:
                misconceptions.append(f"Potential overgeneralization with '{word}'")
        
        return misconceptions

    def _handle_response_finished(self, full_response: str):
        """Handle when the chat response is complete"""
        # Clear the current response buffer
        self.current_ai_response = None
        
        # Add to conversation history
        self.conversation_history.append(full_response)
        
        # Add an extra newline for better spacing
        self.chat_display.insertPlainText("\n")
        
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
        # Track conversation progression
        if self.conversation_state == "introduction":
            self.conversation_state = "basic_understanding"
        elif self.conversation_state == "basic_understanding" and len(self.conversation_history) >= 6:
            self.conversation_state = "exploring"
        elif self.conversation_state == "exploring" and len(self.conversation_history) >= 12:
            self.conversation_state = "connecting_concepts"
            
        # Extract potential topics from user text
        text_lower = user_text.lower()
        key_concepts = ["neural network", "transformer", "attention", "encoder", "decoder", "layer"]
        related_concepts = {
            "neural network": ["layer", "input", "output"],
            "transformer": ["attention", "encoder", "decoder"],
            "attention": ["encoder", "decoder", "weights"],
            "encoder": ["input", "representation"],
            "decoder": ["output", "prediction"],
            "layer": ["neural network", "transformation"]
        }
        
        # Update current concept and track related ones
        for concept in key_concepts:
            if concept in text_lower:
                self.last_concept = concept
                # Store related concepts for future exploration
                if concept in related_concepts:
                    self.related_concepts = related_concepts[concept]
                break

    def _get_teaching_state(self):
        """Determine teaching state based on conversation state"""
        states = {
            "introduction": {"understanding_level": 0, "depth_level": 0},
            "basic_understanding": {"understanding_level": 1, "depth_level": 0},
            "exploring": {"understanding_level": 2, "depth_level": 1},
            "connecting_concepts": {"understanding_level": 3, "depth_level": 2}
        }
        
        current_state = states.get(self.conversation_state, states["introduction"])
        return {
            "current_topic": self.last_concept,
            "understanding_level": current_state["understanding_level"],
            "depth_level": current_state["depth_level"],
            "related_concepts": getattr(self, 'related_concepts', [])
        }

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

    # Set application icon
    app_icon = QIcon("./icons/logo.png")
    app.setWindowIcon(app_icon)

    window = Enso()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()