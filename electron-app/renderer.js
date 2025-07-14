// Renderer process script for Enso Electron app

// DOM Elements
const uploadBtn = document.getElementById('upload-btn');
const modelSelector = document.getElementById('model-selector');
const pageStart = document.getElementById('page-start');
const pageEnd = document.getElementById('page-end');
const statusMessage = document.getElementById('status-message');
const progressContainer = document.querySelector('.progress-container');
const progressBar = document.getElementById('progress-bar');
const documentSection = document.getElementById('document-section');
const chatDisplay = document.getElementById('chat-display');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const boldBtn = document.getElementById('bold-btn');
const italicBtn = document.getElementById('italic-btn');
const bulletBtn = document.getElementById('bullet-btn');
const inputContainer = document.querySelector('.input-container');

// State variables
let documentProcessed = false;
let statusCheckInterval = null;
let conversationHistory = [];
let backendPort = 8080; // Default port, will be updated by main process

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
  // Upload button click handler
  uploadBtn.addEventListener('click', handleUpload);
  
  // Send button click handler
  sendBtn.addEventListener('click', handleSendMessage);
  
  // Enter key in textarea
  userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  });
  
  // Text formatting buttons
  boldBtn.addEventListener('click', () => applyFormatting('**', '**'));
  italicBtn.addEventListener('click', () => applyFormatting('*', '*'));
  bulletBtn.addEventListener('click', () => addBulletPoint());

  // Listen for the backend port from the main process
  window.electronAPI.onBackendPort((event, port) => {
    console.log(`Received backend port: ${port}`);
    backendPort = port;
  });
});

// Handle document upload
async function handleUpload() {
  try {
    // Open file dialog
    const filePath = await window.electronAPI.openFile();
    
    if (!filePath) return; // User canceled
    
    // Get page range
    const startPage = pageStart.value ? parseInt(pageStart.value) : 1;
    const endPage = pageEnd.value ? parseInt(pageEnd.value) : null;
    
    // Disable UI during processing
    uploadBtn.disabled = true;
    userInput.disabled = true;
    sendBtn.disabled = true;
    
    // Show progress bar
    progressContainer.style.display = 'block';
    progressBar.style.width = '0%';
    statusMessage.textContent = 'Starting document processing...';
    
    // Send request to backend using the dynamic port
    await fetch(`http://localhost:${backendPort}/process`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        file_path: filePath,
        start_page: startPage,
        end_page: endPage,
        model_name: modelSelector.value
      }),
    });
    
    // Old direct call (replace with fetch)
    /* await window.electronAPI.backend.processDocument(
      filePath,
      startPage,
      endPage,
      modelSelector.value
    ); */
    
    // Start polling for status
    startStatusPolling();
    
  } catch (error) {
    console.error('Error uploading document:', error);
    statusMessage.textContent = `Error: ${error.message}`;
    uploadBtn.disabled = false;
  }
}

// Poll backend for processing status
function startStatusPolling() {
  // Clear any existing interval
  if (statusCheckInterval) {
    clearInterval(statusCheckInterval);
  }
  
  statusCheckInterval = setInterval(async () => {
    try {
      // Fetch status from backend using the dynamic port
      const response = await fetch(`http://localhost:${backendPort}/status`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const status = await response.json();
      
      // Old direct call (replace with fetch)
      // const status = await window.electronAPI.backend.getStatus();
      
      // Update status message
      statusMessage.textContent = status.message;
      
      // Update progress bar
      if (status.progress > 0) {
        progressBar.style.width = `${status.progress}%`;
      }
      
      // Check if processing is complete or has error
      if (status.status === 'complete') {
        clearInterval(statusCheckInterval);
        progressContainer.style.display = 'none';
        uploadBtn.disabled = false;
        userInput.disabled = false;
        sendBtn.disabled = false;
        documentProcessed = true;
        
        // Hide document section
        documentSection.classList.add('hidden');
        
        // Show the input container with animation
        setTimeout(() => {
          inputContainer.classList.add('visible');
        }, 500);
        
        // Display initial greeting
        appendAIMessage("Hey there! 👋 I'm excited to explore this document with you! Think of me as your curious learning buddy - we're like detectives looking for interesting clues in the text. What caught your eye? Even a simple word or idea is perfect to start our investigation! 🔍");
        
        // Focus on input
        userInput.focus();
      } else if (status.status === 'error') {
        clearInterval(statusCheckInterval);
        progressContainer.style.display = 'none';
        uploadBtn.disabled = false;
        statusMessage.textContent = `Error: ${status.message}`;
      }
      
    } catch (error) {
      console.error('Error checking status:', error);
      clearInterval(statusCheckInterval);
      statusMessage.textContent = `Error checking status: ${error.message}`;
      uploadBtn.disabled = false;
    }
  }, 1000); // Check every second
}

// Handle sending a message
async function handleSendMessage() {
  const message = userInput.value.trim();
  if (!message) return;
  
  // Clear input and disable until response is complete
  userInput.value = '';
  userInput.disabled = true;
  sendBtn.disabled = true;
  
  // Display user message
  appendUserMessage(message);

  // Show "ENSO: Thinking..." message with animation
  const thinkElement = document.createElement('div');
  thinkElement.className = 'message ai-message';
  thinkElement.textContent = 'Thinking...';
  chatDisplay.appendChild(thinkElement);
  
  try {
    // Send message to backend using the dynamic port
    const fetchResponse = await fetch(`http://localhost:${backendPort}/chat`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: message }),
    });

    if (!fetchResponse.ok) {
        throw new Error(`HTTP error! status: ${fetchResponse.status}`);
    }
    const response = await fetchResponse.json();
    
    // Handle streaming response
    if (response.tokens && response.tokens.length > 0) {
      // Remove the "Thinking..." message after its animation completes
      setTimeout(() => {
        chatDisplay.removeChild(thinkElement);
        
        // Start a new AI message
        const messageElement = document.createElement('div');
        messageElement.className = 'message ai-message';
        messageElement.textContent = 'ENSO: ';
        chatDisplay.appendChild(messageElement);
        
        // Stream tokens with slight delay to allow animation to start
        setTimeout(async () => {
          for (const token of response.tokens) {
            messageElement.textContent += token;
            chatDisplay.scrollTop = chatDisplay.scrollHeight;
            const randomInt = Math.floor(Math.random() * 15) + 1;
            await new Promise(resolve => setTimeout(resolve, randomInt));
          }
          messageElement.textContent += '\n';
        }, 100);
      }, 500); // Wait for thinking message animation to complete
      
    } else {
      // Fallback to non-streaming response
      setTimeout(() => {
        chatDisplay.removeChild(thinkElement);
        appendAIMessage(response.response);
      }, 500);
    }
    
  } catch (error) {
    console.error('Error sending message:', error);
    setTimeout(() => {
      chatDisplay.removeChild(thinkElement);
      appendAIMessage("Hmm, I didn't quite catch that. Could you try saying that in a different way? 🤔");
    }, 500);
  } finally {
    // Re-enable input
    userInput.disabled = false;
    sendBtn.disabled = false;
    userInput.focus();
  }
}

// Append a user message to the chat display
function appendUserMessage(text) {
  // Add to conversation history
  conversationHistory.push(text);
  
  // Create message element
  const messageElement = document.createElement('div');
  messageElement.className = 'message user-message';
  messageElement.textContent = `User: ${text}\n`;
  
  // Add to chat display
  chatDisplay.appendChild(messageElement);
  
  // Scroll to bottom after animation starts
  setTimeout(() => {
    chatDisplay.scrollTop = chatDisplay.scrollHeight;
  }, 100);
}

// Append an AI message to the chat display
function appendAIMessage(text) {
  // Add to conversation history
  conversationHistory.push(text);
  
  // Create message element
  const messageElement = document.createElement('div');
  messageElement.className = 'message ai-message';
  messageElement.textContent = `ENSO: ${text}\n`;
  
  // Add to chat display
  chatDisplay.appendChild(messageElement);
  
  // Scroll to bottom after animation starts
  setTimeout(() => {
    chatDisplay.scrollTop = chatDisplay.scrollHeight;
  }, 100);
}

// Text formatting functions
function applyFormatting(prefix, suffix) {
  const start = userInput.selectionStart;
  const end = userInput.selectionEnd;
  const text = userInput.value;
  
  if (start === end) {
    // No selection, insert formatting markers and place cursor between them
    const newText = text.substring(0, start) + prefix + suffix + text.substring(end);
    userInput.value = newText;
    userInput.selectionStart = start + prefix.length;
    userInput.selectionEnd = start + prefix.length;
  } else {
    // Apply formatting to selection
    const selectedText = text.substring(start, end);
    const newText = text.substring(0, start) + prefix + selectedText + suffix + text.substring(end);
    userInput.value = newText;
    userInput.selectionStart = start + prefix.length;
    userInput.selectionEnd = end + prefix.length;
  }
  
  userInput.focus();
}

function addBulletPoint() {
  const start = userInput.selectionStart;
  const text = userInput.value;
  
  // Find the start of the current line
  let lineStart = start;
  while (lineStart > 0 && text[lineStart - 1] !== '\n') {
    lineStart--;
  }
  
  // Insert bullet point at the start of the line
  const newText = text.substring(0, lineStart) + '• ' + text.substring(lineStart);
  userInput.value = newText;
  userInput.selectionStart = start + 2;
  userInput.selectionEnd = start + 2;
  userInput.focus();
}