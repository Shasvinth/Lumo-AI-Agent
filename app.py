"""
Web Interface for Lumo Multilingual Textbook Q&A

This module provides a Flask web application for the Lumo system.
It handles file uploads, query processing, and serving the web interface.
"""

from flask import Flask, request, jsonify, send_from_directory, render_template
import os
import json
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from deep_translator import GoogleTranslator

# Import from the components package
from components import RAGProcessor, process_pdf

# Load environment variables
load_dotenv()

# Initialize the Flask application
app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize RAG processor
rag_processor = None

def translate_to_language(text, target_lang):
    """Translate text to target language"""
    if target_lang == 'en':
        return text
        
    try:
        # Map language codes
        lang_map = {
            'ta': 'ta',  # Tamil
            'si': 'si'   # Sinhala
        }
        google_lang = lang_map.get(target_lang, 'en')
        
        # Translate using Google Translate
        translator = GoogleTranslator(source='auto', target=google_lang)
        translation = translator.translate(text)
        return translation
    except Exception as e:
        print(f"Translation failed: {str(e)}")
        return text

@app.route('/')
def index():
    """Render the index page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle PDF file upload and processing"""
    global rag_processor
    
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        try:
            # First process the PDF into chunks
            chunks_path = os.path.join(UPLOAD_FOLDER, 'chunks.json')
            process_pdf(filepath, chunks_path)
            
            # Initialize RAG processor with the processed chunks
            rag_processor = RAGProcessor()
            
            return jsonify({'success': True})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    return jsonify({'success': False, 'error': 'Invalid file type'})

@app.route('/query', methods=['POST'])
def process_query():
    """Process a question about the textbook"""
    global rag_processor
    
    if not rag_processor:
        return jsonify({'success': False, 'error': 'Please upload a textbook first'})
    
    data = request.json
    query = data.get('query')
    selected_language = data.get('language', 'en')
    
    if not query:
        return jsonify({'success': False, 'error': 'No query provided'})
    
    try:
        # Process the query (will auto-detect language of input)
        result = rag_processor.process_query(
            "user_query",
            query,
            top_k=5
        )
        
        # Force translate answer to selected language regardless of detected language
        if result['Language'] != selected_language:
            answer = translate_to_language(result['Answer'], selected_language)
        else:
            answer = result['Answer']
        
        return jsonify({
            'success': True,
            'answer': answer,
            'sections': result['Sections'],
            'pages': result['Pages'],
            'language': selected_language  # Return selected language, not detected
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)