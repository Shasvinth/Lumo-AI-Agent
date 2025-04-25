// DOM Elements
const pdfUpload = document.getElementById('pdf-upload');
const uploadContainer = document.getElementById('upload-container');
const uploadButton = document.getElementById('upload-button');
const settingsUploadButton = document.getElementById('settings-upload-button');
const languageSelector = document.getElementById('language-selector');
const chatMessages = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const statusText = document.getElementById('status-text');
const statusDot = document.getElementById('status-dot');
const voiceInputButton = document.getElementById('voice-input-button');
const ttsToggle = document.getElementById('tts-toggle');
const darkModeToggle = document.getElementById('dark-mode-toggle');
const currentFileText = document.getElementById('current-file-text');
const currentFileInfo = document.getElementById('current-file-info');
const historyPanel = document.getElementById('history-panel');
const historyButton = document.getElementById('history-button');
const closeHistory = document.getElementById('close-history');
const webSearchIndicator = document.getElementById('web-search-indicator');
const webToggle = document.getElementById('web-toggle');
const textbookToggle = document.getElementById('textbook-toggle');

// Toggle functionality
const toggleSwitch = document.querySelector('.toggle-switch');
const toggleKnob = document.querySelector('.toggle-knob');

// Navigation Elements
const navChat = document.getElementById('nav-chat');
const navSettings = document.getElementById('nav-settings');
const navInfo = document.getElementById('nav-info');
const settingsPanel = document.getElementById('settings-panel');
const aboutPanel = document.getElementById('about-panel');
const closeSettings = document.getElementById('close-settings');
const closeAbout = document.getElementById('close-about');
const overlay = document.getElementById('overlay');

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
let ttsEnabled = false; // Changed to false by default
let darkModeEnabled = false;
let useWebSearch = false;
let useTextbooks = true; // Default to using textbooks
let approvedWebsites = {};

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
uploadButton.addEventListener('click', function() {
    console.log('Upload button clicked');
    pdfUpload.click();
});
settingsUploadButton.addEventListener('click', function() {
    console.log('Settings upload button clicked');
    pdfUpload.click();
});
userInput.addEventListener('input', handleInput);
userInput.addEventListener('keydown', handleKeyPress);
sendButton.addEventListener('click', handleSend);
languageSelector.addEventListener('change', handleLanguageChange);
voiceInputButton.addEventListener('click', toggleSpeechRecognition);
ttsToggle.addEventListener('change', toggleTextToSpeech);
darkModeToggle.addEventListener('click', toggleDarkMode);
historyButton.addEventListener('click', showHistory);
closeHistory.addEventListener('click', hideHistory);

// Manual toggle click handling (in addition to the checkbox change event)
document.querySelector('.toggle-switch').addEventListener('click', function() {
    // Only allow toggle if not disabled
    if (!ttsToggle.disabled) {
        // Toggle the checked state
        ttsToggle.checked = !ttsToggle.checked;
        
        // Manually trigger the change event
        const event = new Event('change');
        ttsToggle.dispatchEvent(event);
    }
});

// Bottom navigation and panel listeners
navChat.addEventListener('click', showChat);
navSettings.addEventListener('click', showSettings);
navInfo.addEventListener('click', showAbout);
closeSettings.addEventListener('click', hideSettings);
closeAbout.addEventListener('click', hideAbout);
overlay.addEventListener('click', hideAllPanels);

// Auto-resize textarea
userInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

// Panel Navigation Functions
function showChat() {
    setActiveNavItem(navChat);
    hideAllPanels();
}

function showSettings() {
    setActiveNavItem(navSettings);
    settingsPanel.classList.add('active');
    overlay.classList.add('active');
    updateFileInfo();
}

function showAbout() {
    setActiveNavItem(navInfo);
    aboutPanel.classList.add('active');
    overlay.classList.add('active');
}

function hideSettings() {
    settingsPanel.classList.remove('active');
    overlay.classList.remove('active');
    setActiveNavItem(navChat);
}

function hideAbout() {
    aboutPanel.classList.remove('active');
    overlay.classList.remove('active');
    setActiveNavItem(navChat);
}

function hideAllPanels() {
    settingsPanel.classList.remove('active');
    aboutPanel.classList.remove('active');
    historyPanel.classList.remove('active');
    overlay.classList.remove('active');
    setActiveNavItem(navChat);
}

function setActiveNavItem(item) {
    // Remove active class from all nav items
    [navChat, navSettings, navInfo].forEach(el => el.classList.remove('active'));
    // Add active class to current item
    item.classList.add('active');
}

function updateFileInfo() {
    if (currentFile) {
        // Update the file info in the settings panel
        currentFileText.textContent = currentFile.name;
        
        // Hide duplicate file info section if it exists
        if (currentFileInfo) {
            currentFileInfo.classList.add('hidden');
        }
        
        // Hide the main upload button in chat view
        uploadContainer.classList.add('hidden');
    } else {
        // Show that no file is uploaded
        currentFileText.textContent = 'No file uploaded';
        
        // Show duplicate file info section if it exists
        if (currentFileInfo) {
            currentFileInfo.classList.remove('hidden');
        }
        
        // Show the main upload button
        uploadContainer.classList.remove('hidden');
    }
}

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
    // Set ttsEnabled based on the current checkbox state
    ttsEnabled = ttsToggle.checked;
    
    // If disabled mid-speech, stop any ongoing speech
    if (!ttsEnabled) {
        speechSynthesis.cancel();
    }
    
    // Update toggle switch visual state for new UI
    updateToggleSwitch();
    
    // Save user preference
    localStorage.setItem('ttsEnabled', ttsEnabled ? 'true' : 'false');
}

function updateToggleSwitch() {
    if (ttsEnabled) {
        toggleSwitch.style.backgroundColor = 'var(--primary-color)';
        toggleKnob.style.left = '24px';
    } else {
        toggleSwitch.style.backgroundColor = '#e5e7eb';
        toggleKnob.style.left = '2px';
    }
    
    // Make sure the checked state matches the enabled state
    ttsToggle.checked = ttsEnabled;
    
    if (ttsToggle.disabled) {
        toggleSwitch.style.opacity = '0.6';
        toggleSwitch.style.cursor = 'not-allowed';
    } else {
        toggleSwitch.style.opacity = '1';
        toggleSwitch.style.cursor = 'pointer';
    }
}

function handleLanguageChange(event) {
    selectedLanguage = event.target.value;
    const languageName = getLanguageName(selectedLanguage);
    
    // Always disable text-to-speech for Tamil and Sinhala
    if (selectedLanguage === 'ta' || selectedLanguage === 'si') {
        ttsEnabled = false;
        ttsToggle.checked = false;
        // Disable the toggle control
        ttsToggle.disabled = true;
    } else {
        // English is available, but respect user's choice
        ttsToggle.disabled = false;
        // Don't change the checked state, keep it as user set it
    }
    
    // Update toggle switch visual state for new UI
    updateToggleSwitch();
    
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
    updateFileInfo(); // Update UI immediately to show the file is being processed
    
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
            
            // Update file info in settings and hide main upload button
            updateFileInfo();
            
            // Make sure we're on the chat tab after successful upload
            showChat();
        } else {
            throw new Error(data.error || 'Failed to process PDF');
        }
    })
    .catch(error => {
        showError(error.message);
        updateStatus('Error', 'error');
        
        // Reset if there was an error
        currentFile = null;
        updateFileInfo();
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

    // Fetch both textbook sources and websites in parallel
    Promise.all([
        // Get textbook sources
        fetch('/selected-sources').then(response => response.json()),
        // Get approved websites
        fetch('/websites').then(response => response.json())
    ])
    .then(([sourcesData, websitesData]) => {
        // Process textbook sources
        const selectedSources = sourcesData.success ? sourcesData.selected_sources : [];
        
        // Process websites
        approvedWebsites = websitesData; // Update global approved websites
        const selectedWebsites = Object.keys(websitesData);
        
        console.log(`Fetched ${selectedSources.length} textbook sources and ${selectedWebsites.length} websites`);
        
        // Send query with both sources
        return selectEndpointAndSendQuery(message, selectedSources, selectedWebsites);
    })
    .catch(error => {
        showError(`Error: ${error.message}. Please try again.`);
        updateStatus('Error', 'error');
        isProcessing = false;
        handleInput();
    });
}

// Helper function to select the appropriate endpoint and send query
function selectEndpointAndSendQuery(message, selectedSources, selectedWebsites) {
    // Determine which endpoint to use based on available sources
    let endpoint, requestData;
    
    console.log(`Selecting endpoint - useTextbooks: ${useTextbooks}, useWebSearch: ${useWebSearch}`);
    console.log(`Selected sources: ${selectedSources.length > 0 ? selectedSources.join(', ') : 'none'}`);
    console.log(`Selected websites: ${selectedWebsites.length > 0 ? selectedWebsites.join(', ') : 'none'}`);
    
    // Check if we have any sources available
    const hasTextbooks = useTextbooks && selectedSources && selectedSources.length > 0;
    const hasWebsites = useWebSearch && selectedWebsites && selectedWebsites.length > 0;
    
    if (!hasTextbooks && !hasWebsites) {
        // No sources available
        showError("No sources available. Please enable at least one textbook or website source.");
        updateStatus('Error', 'error');
        isProcessing = false;
        handleInput();
        return Promise.reject(new Error("No sources available"));
    }
    
    // Always use combined query when possible, regardless of toggle state
    if (hasTextbooks && hasWebsites) {
        console.log("Using combined search - both sources available");
        endpoint = '/combined-query';
        requestData = {
            query: message,
            sources: selectedSources,
            websites: selectedWebsites,
            language: languageSelector.value,
            use_markdown: true
        };
    } else if (hasWebsites) {
        // Web search only
        console.log("Using web search only - no textbooks available");
        endpoint = '/web-query';
        requestData = {
            query: message,
            websites: selectedWebsites,
            language: languageSelector.value,
            use_markdown: true
        };
    } else {
        // Textbook search only
        console.log("Using textbook search only - no websites available");
        endpoint = '/query';
        requestData = {
            query: message,
            sources: selectedSources,
            language: languageSelector.value,
            use_markdown: true
        };
    }
    
    console.log(`Sending request to ${endpoint} with data:`, requestData);
    
    return fetch(endpoint, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.success) {
            // Fix for web sources that might not have pages property
            if (data.web_sources && (!data.pages || data.pages.length === 0)) {
                // Create page references for web sources if missing
                data.pages = data.web_sources.map(source => `Source: ${source.name}`);
            }
            
            // Add assistant message
            addMessage('assistant', data.answer, {
                sections: data.sections || [],
                pages: data.pages || [],
                language: data.language,
                format: data.format || 'plain',
                webSources: data.web_sources
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

// Helper function to process content into a cohesive answer
function processContentForUnifiedResponse(content, metadata) {
    // Remove error messages
    let cleanedContent = content;
    
    // Fix for error messages in web source content
    if (cleanedContent.includes("Error processing query: 'pages'")) {
        cleanedContent = cleanedContent.replace(/From .+?:\s*Error processing query: 'pages'/g, "");
    }
    
    // Check if this is a response with multiple sources
    if (content.match(/^From .+?:/m)) {
        console.log("Processing multi-source content for unified response");
        
        // Extract the sections by source
        const sourceMatches = content.match(/From ([^:]+):\s*([^]*?)(?=(?:From [^:]+:)|$)/g) || [];
        
        // Process each source section
        const validSourceContents = [];
        
        // First pass: extract content from valid sections and identify missing information
        for (const sourceMatch of sourceMatches) {
            // Extract source name and content
            const sourceNameMatch = sourceMatch.match(/From ([^:]+):/);
            const sourceName = sourceNameMatch ? sourceNameMatch[1].trim() : "Unknown";
            
            // Skip if the source has error message or no useful information
            if (sourceMatch.includes("Error processing query") || 
                sourceMatch.toLowerCase().includes("does not contain any information") ||
                sourceMatch.toLowerCase().includes("cannot answer your question") || 
                sourceMatch.toLowerCase().includes("no information about") ||
                sourceMatch.toLowerCase().includes("the provided text does not") ||
                sourceMatch.toLowerCase().includes("this question cannot be answered")) {
                continue;
            }
            
            // Clean up the content
            let sourceContent = sourceMatch
                .replace(/From [^:]+:\s*/, '') // Remove "From source:" prefix
                .trim()
                .replace(/^\s*[-•*]\s*/gm, '') // Remove bullet points at beginning of lines
                .replace(/\n{3,}/g, '\n\n'); // Replace multiple newlines
            
            // If content is not empty after cleaning, add it
            if (sourceContent) {
                validSourceContents.push({
                    sourceName: sourceName,
                    content: sourceContent
                });
            }
        }
        
        // If we have valid content, create a unified response
        if (validSourceContents.length > 0) {
            // Combine all valid content without source attribution
            return validSourceContents.map(item => item.content).join("\n\n").trim();
        } else {
            // No valid content found
            return "I couldn't find relevant information about this in the available sources.";
        }
    }
    
    // If not a multi-source response, return the cleaned content as is
    return cleanedContent;
}

function addMessage(type, content, metadata = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-bubble ${type}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'bubble-content';
    
    // For assistant responses, handle formatting
    if (type === 'assistant') {
        // Process content for a unified response
        const processedContent = processContentForUnifiedResponse(content, metadata);
        
        // Render content as Markdown
        contentDiv.innerHTML = md.render(processedContent);
        contentDiv.classList.add('markdown-content');
    } else {
        // Render as plain text with line breaks
        contentDiv.textContent = content;
    }
    
    messageDiv.appendChild(contentDiv);
    
    // Add metadata (sources, pages, sections) in a more concise way
    if (metadata) {
        // Create citations container for all references
        const citationsDiv = document.createElement('div');
        citationsDiv.className = 'message-metadata';
        
        // Only show useful sections and pages references
        let referencesText = '';
        
        // Add language info if available
        if (metadata.language) {
            referencesText += `Language: ${getLanguageName(metadata.language)}`;
        }
        
        // Add the references div only if there's useful content
        if (referencesText) {
            citationsDiv.textContent = referencesText;
            messageDiv.appendChild(citationsDiv);
        }
        
        // Add web sources if available
        if (metadata.webSources && metadata.webSources.length > 0) {
            const webSourcesDiv = document.createElement('div');
            webSourcesDiv.className = 'web-sources-container';
            
            // Add a heading for web sources
            const sourcesHeading = document.createElement('div');
            sourcesHeading.className = 'sources-heading';
            sourcesHeading.textContent = 'Sources:';
            webSourcesDiv.appendChild(sourcesHeading);
            
            // Add each web source with a link
            metadata.webSources.forEach(source => {
                const sourceDiv = document.createElement('div');
                sourceDiv.className = 'web-source';
                
                // Add icon
                const icon = document.createElement('i');
                icon.className = 'fas fa-globe';
                sourceDiv.appendChild(icon);
                
                // Add source name
                const sourceName = document.createElement('span');
                sourceName.textContent = source.name;
                sourceDiv.appendChild(sourceName);
                
                // Add link if URL is available
                if (source.url) {
                    const link = document.createElement('a');
                    link.href = source.url;
                    link.target = '_blank';
                    link.rel = 'noopener noreferrer';
                    link.textContent = 'Visit source';
                    sourceDiv.appendChild(link);
                }
                
                webSourcesDiv.appendChild(sourceDiv);
            });
            
            messageDiv.appendChild(webSourcesDiv);
        }
    }
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function updateStatus(text, type) {
    statusText.textContent = text;
    statusDot.style.backgroundColor = `var(--${type}-color)`;
    
    // Add box shadow for better visibility in glass theme
    statusDot.style.boxShadow = `0 0 5px var(--${type}-color)`;
}

function showError(message) {
    addMessage('system', `Error: ${message}`);
}

function enableChat() {
    userInput.disabled = false;
    if (currentFile) {
        userInput.placeholder = `Ask a question about ${currentFile.name}...`;
    } else {
        userInput.placeholder = 'Ask a question about the available textbooks...';
    }
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

function showHistory() {
    historyPanel.classList.add('active');
    overlay.classList.add('active');
}

function hideHistory() {
    historyPanel.classList.remove('active');
    overlay.classList.remove('active');
}

// Initialize
userInput.disabled = true;
userInput.placeholder = 'Please upload a textbook first...';
selectedLanguage = languageSelector.value;

// Get user preference for text-to-speech or default to false
ttsEnabled = localStorage.getItem('ttsEnabled') === 'true';
ttsToggle.checked = ttsEnabled;

// Initialize speech recognition and load dark mode preference when the page loads
document.addEventListener('DOMContentLoaded', () => {
    initSpeechRecognition();
    loadDarkModePreference();
    
    // Make sure chat tab is active by default
    setActiveNavItem(navChat);
    
    // Check if currentFile exists and update UI accordingly
    updateFileInfo();
    
    // Initialize the toggle switch to off state
    updateToggleSwitch();
    
    // Initialize web search preference from localStorage or default to false
    useWebSearch = localStorage.getItem('useWebSearch') === 'true';
    if (webToggle) {
        webToggle.checked = useWebSearch;
    }
    
    // Initialize textbook search preference from localStorage or default to true
    useTextbooks = localStorage.getItem('useTextbooks') !== 'false';
    if (textbookToggle) {
        textbookToggle.checked = useTextbooks;
    }
    
    // Check if there are already processed books in the system
    checkExistingBooks();
    
    // Load approved websites
    fetch('/websites')
        .then(response => response.json())
        .then(websites => {
            approvedWebsites = websites;
            console.log("Initialized approved websites:", Object.keys(approvedWebsites));
            updateWebSearchIndicator();
        })
        .catch(error => {
            console.error('Error loading websites:', error);
        });
});

// Function to check if there are already processed books in the system
function checkExistingBooks() {
    fetch('/books')
        .then(response => response.json())
        .then(data => {
            console.log('Checking for existing books:', data);
            if (data.success && data.books && data.books.length > 0) {
                // There are already processed books in the system
                enableChatWithExistingBooks(data.books);
                
                // Update the status to show books are loaded from saved data
                updateStatus('Ready - Loaded from saved data', 'success');
            }
        })
        .catch(error => {
            console.error('Error checking existing books:', error);
        });
}

// Function to enable chat with existing books
function enableChatWithExistingBooks(books) {
    if (!currentFile) {  // Only enable if no file has been uploaded in this session
        console.log('Enabling chat with existing books');
        
        // Create a list of book names with selection status
        const selectedBooks = books.filter(book => book.selected);
        const bookList = selectedBooks.length > 0 ? 
            selectedBooks.map(book => book.name).join(', ') : 
            books.map(book => book.name).join(', ');
        
        // Create message about selection
        let message = '';
        if (selectedBooks.length > 0 && selectedBooks.length < books.length) {
            message = `The following selected textbooks are available: ${selectedBooks.map(book => book.name).join(', ')}`;
            message += `\n\nThere are ${books.length - selectedBooks.length} additional books available in the database.`;
            message += ` You can manage book selections in the Books page.`;
        } else {
            message = `The following textbooks are available: ${bookList}`;
        }
        
        // Enable the chat input
        userInput.disabled = false;
        userInput.placeholder = 'Ask a question about the available textbooks...';
        
        // Update history count
        document.getElementById('history-count').textContent = "0";
        
        // Add a message to inform the user
        addMessage('system', message);
    }
}

// Navigation between panels
document.getElementById('nav-settings').addEventListener('click', function() {
    // Load current source info
    updateSourcesList();
    
    // Load approved websites
    loadApprovedWebsites();
    
    // Show settings panel
    document.getElementById('settings-panel').classList.add('active');
    document.getElementById('about-panel').classList.remove('active');
    document.getElementById('overlay').style.display = 'block';
    
    // Update bottom nav
    document.getElementById('nav-chat').classList.remove('active');
    document.getElementById('nav-settings').classList.add('active');
    document.getElementById('nav-info').classList.remove('active');
});

// Function to load approved websites
function loadApprovedWebsites() {
    const websitesList = document.getElementById('approved-websites-list');
    websitesList.innerHTML = '<p class="loading-websites">Loading approved websites...</p>';
    
    fetch('/websites')
        .then(response => response.json())
        .then(websites => {
            if (Object.keys(websites).length === 0) {
                websitesList.innerHTML = '<p>No approved websites configured.</p>';
                return;
            }
            
            let html = '';
            Object.values(websites).forEach(website => {
                html += `
                <div class="website-item">
                    <div class="website-info">
                        <div class="website-name">${website.name}</div>
                        <div class="website-url">${website.url}</div>
                    </div>
                </div>
                `;
            });
            
            websitesList.innerHTML = html;
        })
        .catch(error => {
            console.error('Error loading websites:', error);
            websitesList.innerHTML = '<p>Error loading websites. Please refresh.</p>';
        });
        
    // Add event listener for initialize websites button
    const initButton = document.getElementById('initialize-websites-btn');
    if (initButton) {
        initButton.addEventListener('click', initializeApprovedWebsites);
    }
}

// Function to initialize approved websites
function initializeApprovedWebsites() {
    const initButton = document.getElementById('initialize-websites-btn');
    const websitesList = document.getElementById('approved-websites-list');
    
    // Disable button and show loading
    initButton.disabled = true;
    initButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Initializing...';
    websitesList.innerHTML = '<p>Initializing approved websites... This may take several minutes.</p>';
    
    // Call the API to initialize websites
    fetch('/initialize-websites', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Show success message
            addMessage('system', data.message);
            
            // Re-enable button
            initButton.disabled = false;
            initButton.innerHTML = '<i class="fas fa-sync"></i> Initialize Approved Websites';
            
            // Reload websites list
            loadApprovedWebsites();
        } else {
            throw new Error(data.error || 'Failed to initialize websites');
        }
    })
    .catch(error => {
        console.error('Error initializing websites:', error);
        websitesList.innerHTML = `<p>Error: ${error.message}</p>`;
        
        // Re-enable button
        initButton.disabled = false;
        initButton.innerHTML = '<i class="fas fa-sync"></i> Initialize Approved Websites';
    });
}

// Initialize web search indicator
function updateWebSearchIndicator() {
    const indicator = document.getElementById('web-search-indicator');
    if (!indicator) return; // Safety check
    
    const statusText = indicator.querySelector('.indicator-status');
    
    if (useWebSearch && (approvedWebsites && Object.keys(approvedWebsites).length > 0)) {
        indicator.classList.add('active');
        statusText.textContent = 'ON';
        console.log("Web search indicator set to ON");
    } else {
        indicator.classList.remove('active');
        statusText.textContent = 'OFF';
        console.log("Web search indicator set to OFF");
    }
    
    // Update the web toggle checkbox to match
    if (webToggle) {
        webToggle.checked = useWebSearch;
    }
}

// Web search indicator click event
document.getElementById('web-search-indicator').addEventListener('click', function() {
    // Only toggle if we have approved websites
    if (approvedWebsites && Object.keys(approvedWebsites).length > 0) {
        // Toggle the useWebSearch state directly
        useWebSearch = !useWebSearch;
        
        // Save preference to localStorage
        localStorage.setItem('useWebSearch', useWebSearch ? 'true' : 'false');
        
        // Update the checkbox
        if (webToggle) {
            webToggle.checked = useWebSearch;
        }
        
        // Update the indicator
        updateWebSearchIndicator();
        console.log("Web search toggled via indicator:", useWebSearch);
    } else {
        alert('You need to add at least one approved website to use web search. Please add a website in the Approved Websites section.');
    }
});

// Add web toggle event listener after initializing all the DOM elements
if (webToggle) {
    webToggle.addEventListener('change', function() {
        useWebSearch = this.checked;
        // Save preference to localStorage
        localStorage.setItem('useWebSearch', useWebSearch ? 'true' : 'false');
        
        updateWebSearchIndicator();
        console.log("Web search toggled:", useWebSearch);

        // If enabling web search but no approved websites, fetch them
        if (useWebSearch && (!approvedWebsites || Object.keys(approvedWebsites).length === 0)) {
            fetch('/websites')
                .then(response => response.json())
                .then(websites => {
                    approvedWebsites = websites;
                    console.log("Loaded approved websites:", Object.keys(approvedWebsites));
                    
                    // If no websites are approved, show warning and disable web search
                    if (Object.keys(approvedWebsites).length === 0) {
                        alert('You need to add at least one approved website to use web search. Please add a website in the Approved Websites section.');
                        this.checked = false;
                        useWebSearch = false;
                        localStorage.setItem('useWebSearch', 'false');
                        updateWebSearchIndicator();
                    }
                })
                .catch(error => {
                    console.error('Error loading websites:', error);
                    alert('Error loading approved websites');
                    this.checked = false;
                    useWebSearch = false;
                    localStorage.setItem('useWebSearch', 'false');
                    updateWebSearchIndicator();
                });
        }
    });
}

// Add textbook toggle event listener
if (textbookToggle) {
    textbookToggle.addEventListener('change', function() {
        useTextbooks = this.checked;
        // Save preference to localStorage
        localStorage.setItem('useTextbooks', useTextbooks ? 'true' : 'false');
        console.log("Textbook search toggled:", useTextbooks);
    });
}