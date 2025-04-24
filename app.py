"""
Web Interface for Lumo Multilingual Textbook Q&A

This module provides a Flask web application for the Lumo system.
It handles file uploads, query processing, and serves the web interface.
Supports multiple PDFs and manages separate embedding stores for each source.
Also supports searching approved web sources for additional information.
"""

from flask import Flask, request, jsonify, send_from_directory, render_template, send_file, Response
import os
import json
import uuid
import shutil
import csv
import io
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
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

# Create web content folder for storing web data
WEB_DATA_FOLDER = os.path.join(UPLOAD_FOLDER, 'web_data')
if not os.path.exists(WEB_DATA_FOLDER):
    os.makedirs(WEB_DATA_FOLDER)

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

# Store approved web sources
APPROVED_WEBSITES = {}

# Track web source processors
WEB_SOURCE_PROCESSORS = {}

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

def fetch_and_process_webpage(url, source_name):
    """Fetch content from a web page and process it for vector storage"""
    try:
        # Send request to URL
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
        
        # Parse HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract text content (remove script and style elements)
        for script in soup(['script', 'style', 'header', 'footer', 'nav']):
            script.extract()
        
        # Get text content
        text = soup.get_text(separator=' ', strip=True)
        
        # Create folder for this web source
        source_folder = os.path.join(WEB_DATA_FOLDER, source_name)
        if not os.path.exists(source_folder):
            os.makedirs(source_folder)
        
        # Process text into chunks (similar to PDF processing but simplified)
        chunks = []
        # Simple chunking by paragraphs (you may want a more sophisticated approach)
        paragraphs = [p for p in text.split('\n') if p.strip()]
        
        for i, paragraph in enumerate(paragraphs):
            if paragraph.strip():
                chunks.append({
                    "chunk_id": f"web_{source_name}_{i}",
                    "text": paragraph,
                    "page": "web",
                    "section": source_name,
                    "source_url": url
                })
        
        # Save chunks to JSON
        chunks_path = os.path.join(source_folder, 'chunks.json')
        with open(chunks_path, 'w') as f:
            json.dump(chunks, f)
        
        # Build vector store
        index_path = os.path.join(source_folder, 'faiss_index.bin')
        processed_chunks_path = os.path.join(source_folder, 'processed_chunks.json')
        
        # Create vector store
        vector_store = build_vector_store(chunks_path, index_path, processed_chunks_path)
        
        return {
            'success': True,
            'path': source_folder,
            'index_path': index_path,
            'chunks_path': processed_chunks_path,
            'chunk_count': len(chunks)
        }
    except Exception as e:
        print(f"Error processing web page {url}: {str(e)}")
        return {'success': False, 'error': str(e)}

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

@app.route('/add-website', methods=['POST'])
def add_website():
    """Add a new approved website to the system"""
    data = request.json
    url = data.get('url')
    name = data.get('name', '')
    description = data.get('description', '')
    
    if not url:
        return jsonify({'success': False, 'error': 'No URL provided'})
    
    # Validate URL format (basic check)
    if not url.startswith(('http://', 'https://')):
        return jsonify({'success': False, 'error': 'Invalid URL format'})
    
    # Generate safe name if not provided
    if not name:
        name = secure_filename(url.replace('https://', '').replace('http://', '').split('/')[0])
    else:
        name = secure_filename(name)
    
    # Don't allow duplicates
    if name in APPROVED_WEBSITES:
        return jsonify({'success': False, 'error': 'Website with this name already exists'})
    
    try:
        # Process the website
        result = fetch_and_process_webpage(url, name)
        
        if not result['success']:
            return jsonify({'success': False, 'error': result['error']})
        
        # Initialize a processor for this web source
        processor = RAGProcessor(result['index_path'], result['chunks_path'])
        WEB_SOURCE_PROCESSORS[name] = processor
        
        # Add to approved websites
        APPROVED_WEBSITES[name] = {
            'url': url,
            'name': name,
            'description': description,
            'path': result['path'],
            'index_path': result['index_path'],
            'chunks_path': result['chunks_path'],
            'chunk_count': result['chunk_count'],
            'added_on': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return jsonify({'success': True, 'websites': APPROVED_WEBSITES})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/websites', methods=['GET'])
def get_websites():
    """Return list of approved websites"""
    return jsonify(APPROVED_WEBSITES)

@app.route('/websites/<name>', methods=['DELETE'])
def remove_website(name):
    """Remove an approved website"""
    if name in APPROVED_WEBSITES:
        try:
            # Delete the website folder
            website_folder = APPROVED_WEBSITES[name]['path']
            shutil.rmtree(website_folder, ignore_errors=True)
            
            # Remove from dictionaries
            del APPROVED_WEBSITES[name]
            if name in WEB_SOURCE_PROCESSORS:
                del WEB_SOURCE_PROCESSORS[name]
            
            return jsonify({'success': True, 'websites': APPROVED_WEBSITES})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    else:
        return jsonify({'success': False, 'error': 'Website not found'})

@app.route('/web-query', methods=['POST'])
def process_web_query():
    """Process a query using approved web sources"""
    global QUERY_HISTORY
    
    data = request.json
    query = data.get('query')
    selected_websites = data.get('websites', [])
    
    if not query:
        return jsonify({'success': False, 'error': 'No query provided'})
    
    if not APPROVED_WEBSITES:
        return jsonify({'success': False, 'error': 'No approved websites available'})
    
    if not selected_websites:
        # If no websites selected, use all available
        selected_websites = list(APPROVED_WEBSITES.keys())
    
    try:
        # Load processors for selected websites
        processors = []
        for website in selected_websites:
            if website in APPROVED_WEBSITES:
                if website not in WEB_SOURCE_PROCESSORS:
                    # Create processor for this website
                    website_data = APPROVED_WEBSITES[website]
                    processor = RAGProcessor(website_data['index_path'], website_data['chunks_path'])
                    WEB_SOURCE_PROCESSORS[website] = processor
                processors.append(WEB_SOURCE_PROCESSORS[website])
        
        if not processors:
            return jsonify({'success': False, 'error': 'No valid websites selected'})
        
        # Process query using all selected web sources
        combined_results = []
        used_sources = set()
        source_urls = {}
        
        for processor in processors:
            query_id = str(uuid.uuid4())
            
            # Get website name from processor
            website_name = None
            for name, proc in WEB_SOURCE_PROCESSORS.items():
                if proc == processor:
                    website_name = name
                    # Store URL for later reference
                    if name in APPROVED_WEBSITES:
                        source_urls[name] = APPROVED_WEBSITES[name]['url']
                    break
            
            # Process query
            result = processor.process_query(
                query_id,
                query,
                top_k=3  # Fewer results per source for web queries
            )
            
            # Only add results if we got an answer
            if result['Answer'].strip():
                combined_results.append(result)
                used_sources.add(website_name if website_name else "Unknown Source")
        
        if not combined_results:
            return jsonify({
                'success': True,
                'answer': "I couldn't find relevant information from the approved web sources.",
                'sections': [],
                'pages': [],
                'language': 'en',
                'web_sources': []
            })
        
        # Combine results (simple approach - concatenate with source attribution)
        combined_answer = ""
        all_sections = set()
        all_sources = []
        
        for i, result in enumerate(combined_results):
            source_name = list(used_sources)[i] if i < len(used_sources) else "Unknown"
            source_url = source_urls.get(source_name, "")
            
            # Add source attribution
            if i > 0:
                combined_answer += "\n\n"
            
            combined_answer += f"{result['Answer']}"
            
            # Collect sections
            if result['Sections']:
                sections = result['Sections'].split(", ")
                for section in sections:
                    if section:
                        all_sections.add(f"{source_name}: {section}")
            
            # Add source reference
            all_sources.append({
                'name': source_name,
                'url': source_url
            })
        
        # Format page references for web sources (they're actually source names)
        page_refs = [f"Source: {s['name']}" for s in all_sources]
        
        # Store in query history
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        history_entry = {
            'timestamp': timestamp,
            'query_id': str(uuid.uuid4()),
            'query': query,
            'answer': combined_answer,
            'sources': ", ".join([s['name'] for s in all_sources]),
            'sections': ", ".join(all_sections),
            'pages': ", ".join(page_refs),
            'language': 'en',
            'type': 'web_query'
        }
        QUERY_HISTORY.append(history_entry)
        
        return jsonify({
            'success': True,
            'answer': combined_answer,
            'sections': list(all_sections),
            'pages': page_refs,
            'language': 'en',
            'web_sources': all_sources
        })
    except Exception as e:
        print(f"Error processing web query: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/combined-query', methods=['POST'])
def process_combined_query():
    """Process a query using both textbooks and approved web sources"""
    global QUERY_HISTORY
    
    data = request.json
    query = data.get('query')
    selected_sources = data.get('sources', [])
    selected_websites = data.get('websites', [])
    
    if not query:
        return jsonify({'success': False, 'error': 'No query provided'})
    
    if not SOURCES and not APPROVED_WEBSITES:
        return jsonify({'success': False, 'error': 'Please upload a textbook or add approved websites first'})
    
    # If nothing selected, use all available
    if not selected_sources and not selected_websites:
        selected_sources = list(SOURCES.keys())
        selected_websites = list(APPROVED_WEBSITES.keys())
    
    try:
        # First process with textbook sources
        textbook_result = None
        if selected_sources and SOURCES:
            textbook_data = request.json.copy()
            textbook_data['sources'] = selected_sources
            textbook_result = process_query().get_json()
        
        # Then process with web sources
        web_result = None
        if selected_websites and APPROVED_WEBSITES:
            web_data = request.json.copy()
            web_data['websites'] = selected_websites
            web_result = process_web_query().get_json()
        
        # Combine results
        combined_answer = ""
        all_sections = []
        all_pages = []
        all_web_sources = []
        
        # Add textbook results
        if textbook_result and textbook_result.get('success'):
            combined_answer += "From textbook sources:\n" + textbook_result['answer']
            all_sections.extend(textbook_result.get('sections', []))
            all_pages.extend(textbook_result.get('pages', []))
        
        # Add web results
        if web_result and web_result.get('success'):
            if combined_answer:
                combined_answer += "\n\nFrom web sources:\n"
            else:
                combined_answer += "From web sources:\n"
            combined_answer += web_result['answer']
            all_sections.extend(web_result.get('sections', []))
            all_pages.extend(web_result.get('pages', []))
            all_web_sources = web_result.get('web_sources', [])
        
        # If no results were found
        if not combined_answer:
            combined_answer = "I couldn't find relevant information from the selected sources."
        
        # Store in query history
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        history_entry = {
            'timestamp': timestamp,
            'query_id': str(uuid.uuid4()),
            'query': query,
            'answer': combined_answer,
            'sources': ", ".join(selected_sources) + " | " + ", ".join(selected_websites),
            'sections': ", ".join(all_sections),
            'pages': ", ".join(all_pages),
            'language': 'en',
            'type': 'combined_query'
        }
        QUERY_HISTORY.append(history_entry)
        
        return jsonify({
            'success': True,
            'answer': combined_answer,
            'sections': all_sections,
            'pages': all_pages,
            'language': 'en',
            'web_sources': all_web_sources
        })
    except Exception as e:
        print(f"Error processing combined query: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)