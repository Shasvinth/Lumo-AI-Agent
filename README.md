# Lumo - Multilingual Textbook Q&A

A multilingual RAG (Retrieval-Augmented Generation) chatbot for answering questions about textbooks in English, Tamil, and Sinhala.

## Features

- **Upload PDF textbooks**: Process and chunk textbooks for semantic search
- **Multilingual support**: Ask questions in English, Tamil, or Sinhala
- **Forced language output**: Get responses in your selected language regardless of input language
- **Smart retrieval**: Find the most relevant textbook sections to answer your questions
- **Source attribution**: See which sections and pages the answers come from
- **Modern chatbot UI**: Clean, responsive interface for easy interaction

## Project Structure

```
├── app.py                    # Main Flask application
├── components/               # Core components
│   ├── __init__.py           # Package initialization
│   ├── embedding_store.py    # Embedding generation and storage
│   ├── pdf_processor.py      # PDF processing module
│   ├── rag_processor.py      # Query processing and response generation
│   └── utils.py              # Utility functions
├── static/                   # Static files
│   ├── css/                  # CSS stylesheets
│   │   └── styles.css        # Main stylesheet
│   ├── images/               # Images and icons
│   │   ├── favicon.ico       # Favicon for the site
│   │   └── Logo.png          # App logo
│   └── js/                   # JavaScript files
│       └── script.js         # Main client-side script
├── templates/                # HTML templates
│   └── index.html            # Main page template
├── uploads/                  # Uploaded files
├── data/                     # Data files
│   └── output/               # Generated data files
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

3. Upload a PDF textbook using the interface

4. Start asking questions in any language (English, Tamil, or Sinhala)

5. Receive answers in your selected language, regardless of input language

## How It Works

1. **PDF Processing**: When you upload a PDF, it's processed into smaller chunks with metadata about sections and page numbers.

2. **Vector Database Building**: Text chunks are converted into vector embeddings using Google's Gemini embedding API and stored in a FAISS index.

3. **Query Processing**: When you ask a question, the system:
   - Detects the language of your input
   - Translates it to English for searching if necessary
   - Finds the most relevant textbook chunks
   - Constructs a prompt with these chunks
   - Generates a response using Gemini API
   - Translates the response to your selected language

## License

[Add your license information here]

## Credits

Developed by [Shasvinth Srikanth](https://github.com/Shasvinth/) and [Resandu Marasinghe](https://github.com/ResanduMarasinghe/)