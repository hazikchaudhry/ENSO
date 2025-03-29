// Preload script to safely expose Electron APIs to the renderer process
const { contextBridge, ipcRenderer } = require('electron');

// Expose a limited set of functionality to the renderer process
contextBridge.exposeInMainWorld('electronAPI', {
  // File dialog functionality
  openFile: () => ipcRenderer.invoke('dialog:openFile'),
  
  // Backend API endpoints
  backend: {
    // Process document
    processDocument: async (filePath, startPage, endPage, modelName) => {
      try {
        const response = await fetch('http://127.0.0.1:5000/process', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            file_path: filePath,
            start_page: startPage,
            end_page: endPage,
            model_name: modelName
          })
        });
        return await response.json();
      } catch (error) {
        console.error('Error processing document:', error);
        throw error;
      }
    },
    
    // Get processing status
    getStatus: async () => {
      try {
        const response = await fetch('http://127.0.0.1:5000/status');
        return await response.json();
      } catch (error) {
        console.error('Error getting status:', error);
        throw error;
      }
    },
    
    // Send chat message
    sendMessage: async (message) => {
      try {
        const response = await fetch('http://127.0.0.1:5000/chat', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ message })
        });
        return await response.json();
      } catch (error) {
        console.error('Error sending message:', error);
        throw error;
      }
    }
  }
});