// Preload script to safely expose Electron APIs to the renderer process
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  openFile: () => ipcRenderer.invoke('dialog:openFile'),
  // Listen for the backend port from the main process
  onBackendPort: (callback) => ipcRenderer.on('set-backend-port', callback)
  // Removed direct backend calls as they are now handled via fetch in renderer.js
});