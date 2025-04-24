"""
Web Interface for Lumo Multilingual Textbook Q&A

This module provides a Flask web application for the Lumo system.
It handles file uploads, query processing, and serves the web interface.
Supports multiple PDFs and manages separate embedding stores for each source.
"""

from flask import Flask, request, jsonify, send_from_directory, render_template, send_file, Response
import os
import json
import uuid
import shutil
import csv
import io
import pandas as pd
from werkzeug.utils import secure_filename
from datetime import datetime
from dotenv import load_dotenv
from deep_translator import GoogleTranslator

# Import from the components package
from components.processors.rag_processor import RAGProcessor
from components.processors.pdf_processor import process_pdf
from components.processors.embedding_store import EmbeddingStore, build_vector_store

# Load environment variables
load_dotenv()

# Initialize the Flask application
app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')

# Configure upload folder and data storage
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Configure data folder for processed files and embeddings
DATA_FOLDER = os.path.join(UPLOAD_FOLDER, 'data')
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

# Create exports folder for CSV files
EXPORTS_FOLDER = os.path.join(UPLOAD_FOLDER, 'exports')
if not os.path.exists(EXPORTS_FOLDER):
    os.makedirs(EXPORTS_FOLDER)

# Track processed sources
SOURCES = {}

# Track source processors to avoid rebuilding embeddings
SOURCE_PROCESSORS = {}

# Store query history for CSV export
QUERY_HISTORY = []

def allowed_file(filename):
    """Check if file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'

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
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    if file and allowed_file(file.filename):
        # Generate a safe filename and source name
        filename = secure_filename(file.filename)
        source_name = os.path.splitext(filename)[0]  # Use filename without extension as source name
        
        # Create a unique folder for this source
        source_folder = os.path.join(DATA_FOLDER, source_name)
        if not os.path.exists(source_folder):
            os.makedirs(source_folder)
        
        # Save the uploaded file
        filepath = os.path.join(source_folder, filename)
        file.save(filepath)
        
        try:
            # Process the PDF into chunks
            chunks_path = os.path.join(source_folder, 'chunks.json')
            process_pdf(filepath, chunks_path)
            
            # Build vector store for this source
            index_path = os.path.join(source_folder, 'faiss_index.bin')
            processed_chunks_path = os.path.join(source_folder, 'processed_chunks.json')
            
            # Create a new vector store for this source
            vector_store = build_vector_store(chunks_path, index_path, processed_chunks_path)
            
            # Initialize a processor for this source
            processor = RAGProcessor(index_path, processed_chunks_path)
            SOURCE_PROCESSORS[source_name] = processor
            
            # Get number of chunks
            with open(processed_chunks_path, 'r') as f:
                chunks = json.load(f)
                chunk_count = len(chunks)
            
            # Add source to the tracking dictionary
            SOURCES[source_name] = {
                'path': source_folder,
                'index_path': index_path,
                'chunks_path': processed_chunks_path,
                'chunk_count': chunk_count
            }
            
            return jsonify({'success': True, 'sources': SOURCES})
        except Exception as e:
            # Clean up on error
            shutil.rmtree(source_folder, ignore_errors=True)
            return jsonify({'success': False, 'error': str(e)})
    
    return jsonify({'success': False, 'error': 'Invalid file type'})

@app.route('/query', methods=['POST'])
def process_query():
    """Process a question about the textbook"""
    global QUERY_HISTORY
    
    data = request.json
    query = data.get('query')
    selected_sources = data.get('sources', [])
    
    # Special case to handle 'hi' or greetings that might trigger old data
    if query.lower().strip() in ['hi', 'hello', 'hey']:
        return jsonify({
            'success': True,
            'answer': f"Hello! I'm Lumo, your textbook assistant. How can I help you today?",
            'sections': [],
            'pages': [],
            'language': 'en'
        })
    
    if not query:
        return jsonify({'success': False, 'error': 'No query provided'})
    
    if not SOURCES:
        return jsonify({'success': False, 'error': 'Please upload a textbook first'})
    
    if not selected_sources:
        # If no sources selected, use all available
        selected_sources = list(SOURCES.keys())
    
    try:
        # Load processors for selected sources
        # If a source doesn't have a processor yet, create one
        processors = []
        for source in selected_sources:
            if source in SOURCES:
                if source not in SOURCE_PROCESSORS:
                    # Create processor for this source
                    source_data = SOURCES[source]
                    processor = RAGProcessor(source_data['index_path'], source_data['chunks_path'])
                    SOURCE_PROCESSORS[source] = processor
                processors.append(SOURCE_PROCESSORS[source])
        
        if not processors:
            return jsonify({'success': False, 'error': 'No valid sources selected'})
        
        # Process the query using first processor for now
        # In a more advanced version, we could combine results from multiple processors
        processor = processors[0]
        query_id = str(uuid.uuid4())
        
        # Process query with existing processor (no new embeddings created)
        result = processor.process_query(
            query_id,
            query,
            top_k=5
        )
        
        # Get source information for references
        source_name = selected_sources[0] if len(selected_sources) == 1 else "Combined Sources"
        
        # Format sections and pages
        sections = result['Sections'].split(', ') if result['Sections'] else []
        
        # Format page references
        page_refs = []
        if result['Pages']:
            pages = result['Pages'].split(', ')
            page_refs = [f"{source_name}: {', '.join(pages)}"]
        
        # Store in query history for CSV export
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        history_entry = {
            'timestamp': timestamp,
            'query_id': query_id,
            'query': query,
            'answer': result['Answer'],
            'sources': ", ".join(selected_sources),
            'sections': result['Sections'],
            'pages': result['Pages'],
            'language': result['Language']
        }
        QUERY_HISTORY.append(history_entry)
        
        return jsonify({
            'success': True,
            'answer': result['Answer'],  # Changed to lowercase to match UI expectations
            'sections': sections,        # Changed to lowercase to match UI expectations
            'pages': page_refs,          # Changed to lowercase to match UI expectations
            'language': result['Language']  # Changed to lowercase to match UI expectations
        })
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/export-csv', methods=['GET'])
def export_csv():
    """Export query history to CSV"""
    if not QUERY_HISTORY:
        return jsonify({'success': False, 'error': 'No query history to export'})
    
    try:
        # Create CSV in memory
        csv_file = io.StringIO()
        fieldnames = ['timestamp', 'query_id', 'query', 'answer', 'sources', 'sections', 'pages', 'language']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        # Write header and rows
        writer.writeheader()
        for entry in QUERY_HISTORY:
            writer.writerow(entry)
        
        # Move pointer to beginning of file
        csv_file.seek(0)
        
        # Set filename with timestamp
        filename = f"lumo_queries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Return CSV as response
        return Response(
            csv_file.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename={filename}'}
        )
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/export-excel', methods=['GET'])
def export_excel():
    """Export query history to Excel"""
    if not QUERY_HISTORY:
        return jsonify({'success': False, 'error': 'No query history to export'})
    
    try:
        # Create DataFrame from query history
        df = pd.DataFrame(QUERY_HISTORY)
        
        # Create Excel file in memory
        excel_file = io.BytesIO()
        with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Query History', index=False)
            
            # Get the xlsxwriter workbook and worksheet objects
            workbook = writer.book
            worksheet = writer.sheets['Query History']
            
            # Add some formatting
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#D7E4BC',
                'border': 1
            })
            
            # Write the column headers with the defined format
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)
                
            # Set column widths
            worksheet.set_column('A:A', 18)  # timestamp
            worksheet.set_column('B:B', 36)  # query_id
            worksheet.set_column('C:C', 40)  # query
            worksheet.set_column('D:D', 60)  # answer
            worksheet.set_column('E:G', 20)  # sources, sections, pages
            worksheet.set_column('H:H', 10)  # language
        
        # Move pointer to beginning of file
        excel_file.seek(0)
        
        # Set filename with timestamp
        filename = f"lumo_queries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        return send_file(
            excel_file,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        print(f"Excel export error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/clear-history', methods=['POST'])
def clear_history():
    """Clear query history"""
    global QUERY_HISTORY
    QUERY_HISTORY = []
    return jsonify({'success': True, 'message': 'Query history cleared'})

@app.route('/sources', methods=['GET'])
def get_sources():
    """Return list of available sources"""
    return jsonify(SOURCES)

@app.route('/sources/<source>', methods=['DELETE'])
def remove_source(source):
    """Remove a source"""
    if source in SOURCES:
        try:
            # Delete the source folder
            source_folder = SOURCES[source]['path']
            shutil.rmtree(source_folder, ignore_errors=True)
            
            # Remove from sources dictionary
            del SOURCES[source]
            
            # Remove from processors dictionary
            if source in SOURCE_PROCESSORS:
                del SOURCE_PROCESSORS[source]
            
            return jsonify({'success': True, 'sources': SOURCES})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    else:
        return jsonify({'success': False, 'error': 'Source not found'})

@app.route('/reset-all', methods=['POST'])
def reset_all_sources():
    """Reset all sources and clear vector stores"""
    global SOURCES, SOURCE_PROCESSORS, QUERY_HISTORY
    
    try:
        # Clear all source folders
        for source_name, source_data in SOURCES.items():
            source_folder = source_data['path']
            try:
                shutil.rmtree(source_folder, ignore_errors=True)
            except Exception:
                pass
        
        # Clear dictionaries
        SOURCES = {}
        SOURCE_PROCESSORS = {}
        QUERY_HISTORY = []
        
        # Also remove any files in the uploads folder that are not in subfolders
        for file in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, file)
            if os.path.isfile(file_path):
                try:
                    os.unlink(file_path)
                except Exception:
                    pass
        
        return jsonify({'success': True, 'message': 'All sources reset successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)