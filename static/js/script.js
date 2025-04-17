// DOM Elements
const pdfUpload = document.getElementById('pdf-upload');
const languageSelector = document.getElementById('language-selector');
const chatMessages = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const statusText = document.getElementById('status-text');
const statusDot = document.getElementById('status-dot');
const voiceInputButton = document.getElementById('voice-input-button');
const ttsToggle = document.getElementById('tts-toggle');
const darkModeToggle = document.getElementById('dark-mode-toggle');

// Initialize markdown-it
const md = window.markdownit({
    html: false,        // Disable HTML tags
    breaks: true,       // Convert '\n' to <br>
    linkify: true,      // Autoconvert URLs to links
    typographer: true,  // Enable smartquotes and other typographic replacements
    highlight: function (str, lang) {
        // Optional syntax highlighting
        return `<code class="language-${lang}">${str}</code>`;
    }
});

// State
let isProcessing = false;
let currentFile = null;
let selectedLanguage = 'en';
let isRecording = false;
let recognition = null;
let speechSynthesis = window.speechSynthesis;
let ttsEnabled = true;
let darkModeEnabled = false;

// Initialize speech recognition if browser supports it
function initSpeechRecognition() {
    if ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window) {
        // Create a speech recognition instance
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognition = new SpeechRecognition();
        
        // Configure speech recognition
        recognition.continuous = false; // Get only one result
        recognition.interimResults = false; // Only return final results
        
        // Set up event handlers
        recognition.onstart = () => {
            isRecording = true;
            voiceInputButton.classList.add('recording');
            updateStatus('Listening...', 'warning');
        };
        
        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            userInput.value = transcript;
            sendButton.disabled = false;
            handleInput();
        };
        
        recognition.onend = () => {
            isRecording = false;
            voiceInputButton.classList.remove('recording');
            updateStatus(currentFile ? 'Ready' : 'Awaiting PDF upload', currentFile ? 'success' : 'warning');
        };
        
        recognition.onerror = (event) => {
            isRecording = false;
            voiceInputButton.classList.remove('recording');
            console.error('Speech recognition error', event.error);
            updateStatus('Error with voice input', 'error');
            setTimeout(() => {
                updateStatus(currentFile ? 'Ready' : 'Awaiting PDF upload', currentFile ? 'success' : 'warning');
            }, 3000);
        };
        
        // Enable the voice input button
        voiceInputButton.disabled = false;
    } else {
        // Browser doesn't support speech recognition
        voiceInputButton.disabled = true;
        voiceInputButton.title = 'Your browser does not support speech recognition';
        console.warn('Speech recognition not supported in this browser');
    }
}

// Function to speak text using text-to-speech
function speakText(text, language) {
    if (!ttsEnabled || !speechSynthesis) return;
    
    // Cancel any ongoing speech
    speechSynthesis.cancel();
    
    // Create a new utterance
    const utterance = new SpeechSynthesisUtterance(text);
    
    // Set language based on the selected language
    switch (language) {
        case 'en':
            utterance.lang = 'en-US';
            break;
        case 'ta':
            utterance.lang = 'ta-IN';
            break;
        case 'si':
            utterance.lang = 'si-LK';
            break;
        default:
            utterance.lang = 'en-US';
    }
    
    // Optional: adjust voice properties
    utterance.rate = 1.0; // Speed: 0.1 to 10
    utterance.pitch = 1.0; // Pitch: 0 to 2
    utterance.volume = 1.0; // Volume: 0 to 1
    
    // Speak the text
    speechSynthesis.speak(utterance);
}

// Event Listeners
pdfUpload.addEventListener('change', handleFileUpload);
userInput.addEventListener('input', handleInput);
userInput.addEventListener('keydown', handleKeyPress);
sendButton.addEventListener('click', handleSend);
languageSelector.addEventListener('change', handleLanguageChange);
voiceInputButton.addEventListener('click', toggleSpeechRecognition);
ttsToggle.addEventListener('change', toggleTextToSpeech);
darkModeToggle.addEventListener('click', toggleDarkMode);

// Auto-resize textarea
userInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

// Functions
function toggleSpeechRecognition() {
    if (!recognition) return;
    
    if (isRecording) {
        recognition.stop();
    } else {
        // Configure recognition language based on the selected language
        switch (selectedLanguage) {
            case 'en':
                recognition.lang = 'en-US';
                break;
            case 'ta':
                recognition.lang = 'ta-IN';
                break;
            case 'si':
                recognition.lang = 'si-LK';
                break;
            default:
                recognition.lang = 'en-US';
        }
        
        recognition.start();
    }
}

function toggleTextToSpeech(event) {
    ttsEnabled = event.target.checked;
    
    // If disabled mid-speech, stop any ongoing speech
    if (!ttsEnabled) {
        speechSynthesis.cancel();
    }
}

function handleLanguageChange(event) {
    selectedLanguage = event.target.value;
    const languageName = getLanguageName(selectedLanguage);
    
    // Automatically disable text-to-speech for Tamil and Sinhala
    if (selectedLanguage === 'ta' || selectedLanguage === 'si') {
        ttsEnabled = false;
        ttsToggle.checked = false;
        // Add a disabled visual state to indicate TTS is not available
        document.querySelector('.toggle-switch').classList.add('disabled');
        // Optionally disable the toggle control
        ttsToggle.disabled = true;
    } else {
        // Re-enable TTS for English
        ttsEnabled = true;
        ttsToggle.checked = true;
        document.querySelector('.toggle-switch').classList.remove('disabled');
        ttsToggle.disabled = false;
    }
    
    // Only show language change message if chat is enabled
    if (!userInput.disabled) {
        let message = `Language changed to ${languageName}. All responses will now be in ${languageName}.`;
        
        // Add info about TTS status
        if (selectedLanguage === 'ta' || selectedLanguage === 'si') {
            message += ` Text-to-speech is not available for ${languageName}.`;
        }
        
        addMessage('system', message);
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
            language: languageSelector.value,
            use_markdown: true // Enable markdown formatting
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
                language: data.language,
                format: data.format || 'plain' // Use the format provided by the server
            });
            updateStatus('Ready', 'success');
            
            // Speak the answer if text-to-speech is enabled
            if (ttsEnabled) {
                speakText(data.answer, data.language);
            }
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
    
    // Check if the content should be rendered as Markdown (only for assistant messages)
    if (type === 'assistant' && metadata && metadata.format === 'markdown') {
        // Render content as Markdown
        contentDiv.innerHTML = md.render(content);
        contentDiv.classList.add('markdown-content');
    } else {
        // Render as plain text with line breaks
        contentDiv.textContent = content;
    }
    
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

function toggleDarkMode() {
    // Update darkModeEnabled state
    darkModeEnabled = !darkModeEnabled;
    
    // Toggle dark mode class on root element
    if (darkModeEnabled) {
        document.documentElement.classList.add('dark-mode');
        localStorage.setItem('darkMode', 'enabled');
        // Change icon to sun when in dark mode
        document.querySelector('#dark-mode-toggle i').classList.remove('fa-moon');
        document.querySelector('#dark-mode-toggle i').classList.add('fa-sun');
    } else {
        document.documentElement.classList.remove('dark-mode');
        localStorage.setItem('darkMode', 'disabled');
        // Change icon to moon when in light mode
        document.querySelector('#dark-mode-toggle i').classList.remove('fa-sun');
        document.querySelector('#dark-mode-toggle i').classList.add('fa-moon');
    }
}

// Function to load user preference for dark mode
function loadDarkModePreference() {
    const darkMode = localStorage.getItem('darkMode');
    
    if (darkMode === 'enabled') {
        darkModeEnabled = true;
        document.documentElement.classList.add('dark-mode');
        // Update icon to sun when loading in dark mode
        document.querySelector('#dark-mode-toggle i').classList.remove('fa-moon');
        document.querySelector('#dark-mode-toggle i').classList.add('fa-sun');
    }
}

// Initialize
userInput.disabled = true;
userInput.placeholder = 'Please upload a textbook first...';
selectedLanguage = languageSelector.value;
ttsEnabled = ttsToggle.checked;

// Initialize speech recognition and load dark mode preference when the page loads
document.addEventListener('DOMContentLoaded', () => {
    initSpeechRecognition();
    loadDarkModePreference();
});