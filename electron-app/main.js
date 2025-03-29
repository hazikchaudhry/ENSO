const { app, BrowserWindow, dialog, ipcMain } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');

// Keep a global reference of the window object to prevent garbage collection
let mainWindow;

// Keep a reference to the Python backend process
let backendProcess = null;

// Start the Python backend server
function startBackendServer() {
  console.log('Starting Python backend server...');
  
  // Path to the Python executable (using the system Python)
  const pythonExecutable = 'python3';
  
  // Path to the backend server script
  const scriptPath = path.join(__dirname, 'backend', 'server.py');
  
  // Check if the server script exists
  if (!fs.existsSync(scriptPath)) {
    console.error(`Backend server script not found at ${scriptPath}`);
    dialog.showErrorBox('Backend Error', 'Backend server script not found.');
    return null;
  }
  
  // Spawn the Python process
  const process = spawn(pythonExecutable, [scriptPath]);
  
  // Handle process events
  process.stdout.on('data', (data) => {
    console.log(`Backend stdout: ${data}`);
  });
  
  process.stderr.on('data', (data) => {
    console.error(`Backend stderr: ${data}`);
  });
  
  process.on('close', (code) => {
    console.log(`Backend process exited with code ${code}`);
    backendProcess = null;
  });
  
  return process;
}

// Handle file dialog open
ipcMain.handle('dialog:openFile', async () => {
  const { canceled, filePaths } = await dialog.showOpenDialog({
    properties: ['openFile'],
    filters: [
      { name: 'Documents', extensions: ['pdf', 'docx', 'txt'] }
    ]
  });
  
  if (!canceled) {
    return filePaths[0];
  }
  return null;
});

// Create the main application window
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1000,
    height: 800,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false
    },
    icon: path.join(__dirname, 'assets', 'icons', 'logo.png')
  });
  
  // Load the index.html file
  mainWindow.loadFile('index.html');
  
  // Open DevTools in development mode
  if (process.env.NODE_ENV === 'development') {
    mainWindow.webContents.openDevTools();
  }
  
  // Handle window close
  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// Initialize the app when Electron is ready
app.whenReady().then(() => {
  // Start the backend server
  backendProcess = startBackendServer();
  
  // Create the main window
  createWindow();


  
  // Re-create window on macOS when clicking on dock icon
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

// Quit the app when all windows are closed (except on macOS)
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// Clean up before quitting
app.on('before-quit', () => {
  // Kill the backend process if it's running
  if (backendProcess) {
    console.log('Stopping Python backend server...');
    backendProcess.kill();
    backendProcess = null;
  }
});