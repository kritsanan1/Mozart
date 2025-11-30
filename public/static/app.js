// OMR Web Application JavaScript

let currentFile = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    setupEventListeners();
    simulateProgress();
});

// Setup event listeners
function setupEventListeners() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const processBtn = document.getElementById('processBtn');
    const clearBtn = document.getElementById('clearBtn');
    const clearResultsBtn = document.getElementById('clearResultsBtn');
    const downloadBtn = document.getElementById('downloadBtn');
    const newProcessBtn = document.getElementById('newProcessBtn');

    // File upload events
    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', handleDragOver);
    dropZone.addEventListener('drop', handleDrop);
    fileInput.addEventListener('change', handleFileSelect);

    // Button events
    processBtn.addEventListener('click', processImage);
    clearBtn.addEventListener('click', clearUpload);
    downloadBtn.addEventListener('click', downloadResults);
    newProcessBtn.addEventListener('click', resetForNewProcess);
}

// Handle drag over
function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    e.dataTransfer.dropEffect = 'copy';
    
    const dropZone = document.getElementById('dropZone');
    dropZone.classList.add('border-indigo-500', 'bg-indigo-50');
}

// Handle file drop
function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    
    const dropZone = document.getElementById('dropZone');
    dropZone.classList.remove('border-indigo-500', 'bg-indigo-50');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

// Handle file selection
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

// Handle file processing
function handleFile(file) {
    // Validate file type
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'application/pdf'];
    if (!validTypes.includes(file.type)) {
        showNotification('Please select a valid image file (JPG, PNG, or PDF)', 'error');
        return;
    }

    // Validate file size (max 10MB)
    const maxSize = 10 * 1024 * 1024; // 10MB
    if (file.size > maxSize) {
        showNotification('File size must be less than 10MB', 'error');
        return;
    }

    currentFile = file;
    displayPreview(file);
}

// Display file preview
function displayPreview(file) {
    const previewArea = document.getElementById('previewArea');
    const previewImage = document.getElementById('previewImage');
    
    if (file.type === 'application/pdf') {
        // For PDF, show a placeholder
        previewImage.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjI1MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxyZWN0IHdpZHRoPSIxMDAlIiBoZWlnaHQ9IjEwMCUiIGZpbGw9IiNmMGYwZjAiLz4KICAgIDx0ZXh0IHg9IjUwJSIgeT0iNTAlIiBmb250LXNpemU9IjE4IiBmaWxsPSIjNjY2IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBkeT0iLjNlbSI+UERGIEZpbGU8L3RleHQ+Cjwvc3ZnPg==';
        previewArea.classList.remove('hidden');
    } else {
        // For images, show preview
        const reader = new FileReader();
        reader.onload = function(e) {
            previewImage.src = e.target.result;
            previewArea.classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    }
}

// Process the uploaded image
async function processImage() {
    if (!currentFile) {
        showNotification('Please select an image first', 'error');
        return;
    }

    const formData = new FormData();
    formData.append('image', currentFile);

    // Show processing section
    document.getElementById('processingSection').classList.remove('hidden');
    document.getElementById('resultsSection').classList.add('hidden');

    try {
        const response = await axios.post('/api/process', formData, {
            headers: {
                'Content-Type': 'multipart/form-data'
            }
        });

        if (response.data.success) {
            displayResults(response.data);
        } else {
            throw new Error(response.data.error || 'Processing failed');
        }
    } catch (error) {
        console.error('Processing error:', error);
        showNotification('Processing failed: ' + (error.response?.data?.error || error.message), 'error');
        document.getElementById('processingSection').classList.add('hidden');
    }
}

// Display processing results
function displayResults(data) {
    document.getElementById('processingSection').classList.add('hidden');
    document.getElementById('resultsSection').classList.remove('hidden');

    // Update counts and scores
    document.getElementById('notesCount').textContent = data.result.notes.length;
    document.getElementById('symbolsCount').textContent = data.result.symbols.length;
    document.getElementById('confidenceScore').textContent = Math.round(data.result.confidence * 100) + '%';

    // Display detailed results
    const detailedResults = document.getElementById('detailedResults');
    detailedResults.innerHTML = '';

    // Add notes
    if (data.result.notes.length > 0) {
        const notesDiv = document.createElement('div');
        notesDiv.className = 'bg-green-100 rounded p-3 mb-2';
        notesDiv.innerHTML = `
            <strong>Notes:</strong> ${data.result.notes.join(', ')}
        `;
        detailedResults.appendChild(notesDiv);
    }

    // Add symbols
    if (data.result.symbols.length > 0) {
        const symbolsDiv = document.createElement('div');
        symbolsDiv.className = 'bg-blue-100 rounded p-3 mb-2';
        symbolsDiv.innerHTML = `
            <strong>Symbols:</strong> ${data.result.symbols.join(', ')}
        `;
        detailedResults.appendChild(symbolsDiv);
    }

    // Add confidence info
    const confidenceDiv = document.createElement('div');
    confidenceDiv.className = 'bg-purple-100 rounded p-3';
    confidenceDiv.innerHTML = `
        <strong>Processing Confidence:</strong> ${Math.round(data.result.confidence * 100)}%
    `;
    detailedResults.appendChild(confidenceDiv);
}

// Clear current upload
clearUpload = function() {
    currentFile = null;
    document.getElementById('previewArea').classList.add('hidden');
    document.getElementById('fileInput').value = '';
    document.getElementById('resultsSection').classList.add('hidden');
    document.getElementById('processingSection').classList.add('hidden');
}

// Reset for new processing
function resetForNewProcess() {
    clearUpload();
    document.getElementById('resultsSection').classList.add('hidden');
}

// Download results
function downloadResults() {
    // Create a simple text report
    const notes = document.getElementById('notesCount').textContent;
    const symbols = document.getElementById('symbolsCount').textContent;
    const confidence = document.getElementById('confidenceScore').textContent;
    
    const report = `Mozart OMR Processing Results
============================
Processed File: ${currentFile?.name || 'Unknown'}
Processing Date: ${new Date().toLocaleString()}

Results:
- Notes Detected: ${notes}
- Symbols Found: ${symbols}
- Confidence Score: ${confidence}

Generated by Mozart OMR System
`;
    
    const blob = new Blob([report], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'omr_results.txt';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Show notification
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `fixed top-4 right-4 p-4 rounded-lg shadow-lg z-50 ${
        type === 'error' ? 'bg-red-500 text-white' : 
        type === 'success' ? 'bg-green-500 text-white' : 
        'bg-blue-500 text-white'
    }`;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.remove();
    }, 3000);
}

// Simulate progress for demo purposes
function simulateProgress() {
    const progressBar = document.getElementById('progressBar');
    let progress = 0;
    
    const interval = setInterval(() => {
        progress += Math.random() * 15;
        if (progress > 90) progress = 90;
        
        if (progressBar) {
            progressBar.style.width = progress + '%';
        }
        
        if (progress >= 90) {
            clearInterval(interval);
        }
    }, 200);
}

// Export functions for global use
window.clearUpload = clearUpload;