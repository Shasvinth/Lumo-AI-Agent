// DOM Elements
const pdfUpload = document.getElementById('pdf-upload');
const languageSelector = document.getElementById('language-selector');
const chatMessages = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const statusText = document.getElementById('status-text');
const statusDot = document.getElementById('status-dot');

// State
let isProcessing = false;
let currentFile = null;
let selectedLanguage = 'en';

// Event Listeners
pdfUpload.addEventListener('change', handleFileUpload);
userInput.addEventListener('input', handleInput);
userInput.addEventListener('keydown', handleKeyPress);
sendButton.addEventListener('click', handleSend);
languageSelector.addEventListener('change', handleLanguageChange);

// Auto-resize textarea
userInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

// Functions
function handleLanguageChange(event) {
    selectedLanguage = event.target.value;
    const languageName = getLanguageName(selectedLanguage);
    
    // Only show language change message if chat is enabled
    if (!userInput.disabled) {
        addMessage('system', `Language changed to ${languageName}. All responses will now be in ${languageName}.`);
    }
}

function getLanguageName(code) {
    const languages = {
        'en': 'English',
        'ta': 'Tamil',
        'si': 'Sinhala'
    };
    return languages[code] || 'English';
}

function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    if (file.type !== 'application/pdf') {
        showError('Please upload a PDF file');
        return;
    }
    
    currentFile = file;
    updateStatus('Processing PDF...', 'warning');
    
    // Create FormData
    const formData = new FormData();
    formData.append('file', file);
    
    // Upload file
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.success) {
            updateStatus('Ready', 'success');
            enableChat();
            const languageName = getLanguageName(selectedLanguage);
            addMessage('system', `PDF processed successfully! You can now ask questions about ${currentFile.name} in ${languageName}.`);
        } else {
            throw new Error(data.error || 'Failed to process PDF');
        }
    })
    .catch(error => {
        showError(error.message);
        updateStatus('Error', 'error');
    });
}

function handleInput() {
    sendButton.disabled = !userInput.value.trim() || isProcessing;
}

function handleKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        handleSend();
    }
}

function handleSend() {
    const message = userInput.value.trim();
    if (!message || isProcessing) return;
    
    // Add user message
    addMessage('user', message);
    
    // Clear input
    userInput.value = '';
    userInput.style.height = 'auto';
    sendButton.disabled = true;
    
    // Update status
    updateStatus('Processing...', 'warning');
    isProcessing = true;
    
    // Send to backend
    fetch('/query', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            query: message,
            language: languageSelector.value
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.success) {
            // Add assistant message
            addMessage('assistant', data.answer, {
                sections: data.sections,
                pages: data.pages,
                language: data.language
            });
            updateStatus('Ready', 'success');
        } else {
            throw new Error(data.error || 'Failed to process query');
        }
    })
    .catch(error => {
        showError(`Error: ${error.message}. Please try again.`);
        updateStatus('Error', 'error');
    })
    .finally(() => {
        isProcessing = false;
        handleInput();
    });
}

function addMessage(type, content, metadata = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = content;
    messageDiv.appendChild(contentDiv);
    
    if (metadata) {
        const metadataDiv = document.createElement('div');
        metadataDiv.className = 'message-metadata';
        
        let metadataText = '';
        if (metadata.sections) {
            metadataText += `Sections: ${metadata.sections}`;
        }
        
        if (metadata.pages) {
            if (metadataText) metadataText += ' | ';
            metadataText += `Pages: ${metadata.pages}`;
        }
        
        if (metadata.language) {
            if (metadataText) metadataText += ' | ';
            metadataText += `Language: ${getLanguageName(metadata.language)}`;
        }
        
        metadataDiv.textContent = metadataText;
        messageDiv.appendChild(metadataDiv);
    }
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function updateStatus(text, type) {
    statusText.textContent = text;
    statusDot.style.backgroundColor = `var(--${type}-color)`;
}

function showError(message) {
    addMessage('system', `Error: ${message}`);
}

function enableChat() {
    userInput.disabled = false;
    userInput.placeholder = `Ask a question about ${currentFile.name}...`;
}

// Initialize
userInput.disabled = true;
userInput.placeholder = 'Please upload a textbook first...';
selectedLanguage = languageSelector.value;