# Lumo - Multilingual Textbook & Web Q&A

A multilingual RAG (Retrieval-Augmented Generation) chatbot for answering questions about textbooks and approved web sources in English, Tamil, and Sinhala.

## Features

- **Upload PDF textbooks**: Process and chunk textbooks for semantic search
- **Web search capability**: Query information from approved web sources
- **Multilingual support**: Ask questions in English, Tamil, or Sinhala
- **Forced language output**: Get responses in your selected language regardless of input language
- **Smart retrieval**: Find the most relevant textbook sections and web sources to answer your questions
- **Unified responses**: Get cohesive, single-voice answers that combine information from multiple sources
- **Source attribution**: See which textbook sections, pages, and web sources the answers come from
- **Modern chatbot UI**: Clean, responsive interface for easy interaction

## Project Structure

```
├── app.py                    # Main Flask application
├── components/               # Core components
│   ├── __init__.py           # Package initialization
│   ├── processors/           # Data processing modules
│   │   ├── embedding_store.py  # Embedding generation and storage
│   │   ├── pdf_processor.py    # PDF processing module
│   │   └── rag_processor.py    # Query processing and response generation
│   └── utils/                # Utility modules
│       └── utils.py          # Utility functions
├── static/                   # Static files
│   ├── css/                  # CSS stylesheets
│   │   └── styles.css        # Main stylesheet
│   ├── images/               # Images and icons
│   │   ├── favicon.ico       # Favicon for the site
│   │   └── Logo.png          # App logo
│   └── js/                   # JavaScript files
│       └── script.js         # Main client-side script
├── templates/                # HTML templates
│   ├── index.html            # Main page template
│   ├── books.html            # Book management template
│   └── websites.html         # Website management template
├── uploads/                  # Uploaded and processed files
│   ├── data/                 # Textbook data and indices
│   ├── web_data/             # Web source data and indices
│   ├── sources.json          # Available textbook sources
│   ├── selected_sources.json # Selected textbook sources
│   └── websites.json         # Approved websites configuration
└── requirements.txt          # Project dependencies
```

## Requirements

- Python 3.10 or higher
- Flask
- Google Gemini API key
- PyPDF2
- FAISS for vector storage
- langdetect
- deep-translator

## Setup

1. Clone the repository:

   ```
   git clone <repository-url>
   cd <Project Directory>
   ```

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   ```
   export GEMINI_API_KEY="your-gemini-api-key"
   ```
   Or create a `.env` file with:
   ```
   GEMINI_API_KEY=your-gemini-api-key
   ```

## Running the Application

1. Start the Flask server:

   ```
   python app.py
   ```

2. Open your browser and navigate to:

   ```
   http://localhost:5000
   ```

3. Upload a PDF textbook using the interface or add approved websites

4. Start asking questions in any language (English, Tamil, or Sinhala)

5. Receive answers in your selected language, with information from textbooks and/or web sources

## How It Works

1. **Content Processing**: 
   - PDF textbooks are processed into smaller chunks with metadata about sections and page numbers
   - Web sources are processed and stored with source attribution

2. **Vector Database Building**: 
   - Text chunks are converted into vector embeddings using Google's Gemini embedding API
   - Embeddings are stored in FAISS indices for fast semantic search

3. **Query Processing**: When you ask a question, the system:
   - Detects the language of your input
   - Translates it to English for searching if necessary
   - Finds the most relevant chunks from textbooks and/or web sources
   - Constructs a prompt with these chunks
   - Generates a cohesive response using Gemini API
   - Translates the response to your selected language if needed

4. **Web Search**: 
   - Toggle web search on/off in the interface
   - Combined search uses both textbooks and web sources
   - Results display citations to original sources

## License

[Add your license information here]

## Credits

Developed by [Shasvinth Srikanth](https://github.com/Shasvinth/) and [Resandu Marasinghe](https://github.com/ResanduMarasinghe/)