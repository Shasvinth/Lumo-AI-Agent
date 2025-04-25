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

# File to store persistent data
SOURCES_FILE = os.path.join(UPLOAD_FOLDER, 'sources.json')
SELECTED_SOURCES_FILE = os.path.join(UPLOAD_FOLDER, 'selected_sources.json')
WEBSITES_FILE = os.path.join(UPLOAD_FOLDER, 'websites.json')

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

# Track user's selected sources
SELECTED_SOURCES = []

def save_sources_to_disk():
    """Save SOURCES dictionary to disk for persistence"""
    try:
        with open(SOURCES_FILE, 'w') as f:
            json.dump(SOURCES, f, indent=2)
        print(f"Saved {len(SOURCES)} sources to {SOURCES_FILE}")
        return True
    except Exception as e:
        print(f"Error saving sources to disk: {str(e)}")
        return False

def load_sources_from_disk():
    """Load SOURCES dictionary from disk"""
    global SOURCES
    try:
        if os.path.exists(SOURCES_FILE):
            with open(SOURCES_FILE, 'r') as f:
                SOURCES = json.load(f)
            print(f"Loaded {len(SOURCES)} sources from {SOURCES_FILE}")
            return True
        return False
    except Exception as e:
        print(f"Error loading sources from disk: {str(e)}")
        return False

def save_selected_sources_to_disk():
    """Save SELECTED_SOURCES list to disk for persistence"""
    try:
        with open(SELECTED_SOURCES_FILE, 'w') as f:
            json.dump(SELECTED_SOURCES, f, indent=2)
        print(f"Saved {len(SELECTED_SOURCES)} selected sources to {SELECTED_SOURCES_FILE}")
        return True
    except Exception as e:
        print(f"Error saving selected sources to disk: {str(e)}")
        return False

def load_selected_sources_from_disk():
    """Load SELECTED_SOURCES list from disk"""
    global SELECTED_SOURCES
    try:
        if os.path.exists(SELECTED_SOURCES_FILE):
            with open(SELECTED_SOURCES_FILE, 'r') as f:
                SELECTED_SOURCES = json.load(f)
            print(f"Loaded {len(SELECTED_SOURCES)} selected sources from {SELECTED_SOURCES_FILE}")
            return True
        return False
    except Exception as e:
        print(f"Error loading selected sources from disk: {str(e)}")
        return False

def save_websites_to_disk():
    """Save APPROVED_WEBSITES dictionary to disk for persistence"""
    try:
        with open(WEBSITES_FILE, 'w') as f:
            json.dump(APPROVED_WEBSITES, f, indent=2)
        print(f"Saved {len(APPROVED_WEBSITES)} websites to {WEBSITES_FILE}")
        return True
    except Exception as e:
        print(f"Error saving websites to disk: {str(e)}")
        return False

def load_websites_from_disk():
    """Load APPROVED_WEBSITES dictionary from disk"""
    global APPROVED_WEBSITES
    try:
        if os.path.exists(WEBSITES_FILE):
            with open(WEBSITES_FILE, 'r') as f:
                APPROVED_WEBSITES = json.load(f)
            print(f"Loaded {len(APPROVED_WEBSITES)} websites from {WEBSITES_FILE}")
            return True
        return False
    except Exception as e:
        print(f"Error loading websites from disk: {str(e)}")
        return False

def verify_data_integrity():
    """
    Verify data integrity by checking for missing files/folders
    and fixing any inconsistencies in loaded data.
    """
    global SOURCES, SELECTED_SOURCES, APPROVED_WEBSITES
    
    print("Verifying data integrity...")
    
    # Check that all SOURCES have valid paths
    sources_to_remove = []
    for source_name, source_data in SOURCES.items():
        # Check if required files exist
        if not os.path.exists(source_data.get('chunks_path', '')) or \
           not os.path.exists(source_data.get('index_path', '')):
            print(f"Warning: Source '{source_name}' has missing files. Marking for removal.")
            sources_to_remove.append(source_name)
    
    # Remove invalid sources
    for source_name in sources_to_remove:
        SOURCES.pop(source_name, None)
        if source_name in SELECTED_SOURCES:
            SELECTED_SOURCES.remove(source_name)
    
    if sources_to_remove:
        print(f"Removed {len(sources_to_remove)} invalid sources.")
        save_sources_to_disk()
        save_selected_sources_to_disk()
    
    # Verify SELECTED_SOURCES only contains valid sources
    valid_selected = [s for s in SELECTED_SOURCES if s in SOURCES]
    if len(valid_selected) != len(SELECTED_SOURCES):
        print(f"Cleaning up selected sources list: {len(SELECTED_SOURCES)} -> {len(valid_selected)}")
        SELECTED_SOURCES = valid_selected
        save_selected_sources_to_disk()
        
    # Check that all APPROVED_WEBSITES have valid paths
    websites_to_remove = []
    for website_name, website_data in APPROVED_WEBSITES.items():
        # Check if required files exist
        if not os.path.exists(website_data.get('chunks_path', '')) or \
           not os.path.exists(website_data.get('index_path', '')):
            print(f"Warning: Website '{website_name}' has missing files. Marking for removal.")
            websites_to_remove.append(website_name)
    
    # Remove invalid websites
    for website_name in websites_to_remove:
        APPROVED_WEBSITES.pop(website_name, None)
    
    if websites_to_remove:
        print(f"Removed {len(websites_to_remove)} invalid websites.")
        save_websites_to_disk()
    
    print("Data integrity verification completed.")

def initialize_processors():
    """Initialize processors for all sources"""
    global SOURCE_PROCESSORS, WEB_SOURCE_PROCESSORS
    
    # Initialize processors for book sources
    for source_name, source_data in SOURCES.items():
        if source_name not in SOURCE_PROCESSORS:
            try:
                processor = RAGProcessor(source_data['index_path'], source_data['chunks_path'])
                SOURCE_PROCESSORS[source_name] = processor
                print(f"Initialized processor for source: {source_name}")
            except Exception as e:
                print(f"Error initializing processor for {source_name}: {str(e)}")
    
    # Initialize processors for web sources
    for website_name, website_data in APPROVED_WEBSITES.items():
        if website_name not in WEB_SOURCE_PROCESSORS:
            try:
                processor = RAGProcessor(website_data['index_path'], website_data['chunks_path'])
                WEB_SOURCE_PROCESSORS[website_name] = processor
                print(f"Initialized processor for website: {website_name}")
            except Exception as e:
                print(f"Error initializing processor for website {website_name}: {str(e)}")

def initialize_approved_websites():
    """Initialize the system with pre-approved websites for search"""
    approved_sites = [
        {"url": "https://kids.nationalgeographic.com/history/article/wright-brothers", "name": "natgeo_wright_brothers"},
        {"url": "https://en.wikipedia.org/wiki/Wright_Flyer", "name": "wiki_wright_flyer"},
        {"url": "https://airandspace.si.edu/collection-objects/1903-wright-flyer/nasm_A19610048000", "name": "airandspace_wright_flyer"},
        {"url": "https://en.wikipedia.org/wiki/Wright_brothers", "name": "wiki_wright_brothers"},
        {"url": "https://spacecenter.org/a-look-back-at-the-wright-brothers-first-flight/", "name": "spacecenter_wright_first_flight"},
        {"url": "https://udithadevapriya.medium.com/a-history-of-education-in-sri-lanka-bf2d6de2882c", "name": "medium_srilanka_education"},
        {"url": "https://en.wikipedia.org/wiki/Education_in_Sri_Lanka", "name": "wiki_srilanka_education"},
        {"url": "https://thuppahis.com/2018/05/16/the-earliest-missionary-english-schools-challenging-shirley-somanader/", "name": "thuppahis_missionary_schools"},
        {"url": "https://www.elivabooks.com/pl/book/book-6322337660", "name": "elivabooks_6322337660"},
        {"url": "https://quizgecko.com/learn/christian-missionary-organizations-in-sri-lanka-bki3tu", "name": "quizgecko_missionary_orgs"},
        {"url": "https://en.wikipedia.org/wiki/Mahaweli_Development_programme", "name": "wiki_mahaweli_development"},
        {"url": "https://www.cmg.lk/largest-irrigation-project", "name": "cmg_irrigation_project"},
        {"url": "https://mahaweli.gov.lk/Corporate%20Plan%202019%20-%202023.pdf", "name": "mahaweli_corporate_plan"},
        {"url": "https://www.sciencedirect.com/science/article/pii/S0016718524002082", "name": "sciencedirect_S0016718524002082"},
        {"url": "https://www.sciencedirect.com/science/article/pii/S2405844018381635", "name": "sciencedirect_S2405844018381635"},
        {"url": "https://www.britannica.com/story/did-marie-antoinette-really-say-let-them-eat-cake", "name": "britannica_marie_antoinette"},
        {"url": "https://genikuckhahn.blog/2023/06/10/marie-antoinette-and-the-infamous-phrase-did-she-really-say-let-them-eat-cake/", "name": "genikuckhahn_marie_antoinette"},
        {"url": "https://www.instagram.com/mottahedehchina/p/Cx07O8XMR8U/?hl=en", "name": "instagram_mottahedehchina"},
        {"url": "https://www.reddit.com/r/HistoryMemes/comments/rqgcjs/let_them_eat_cake_is_the_most_famous_quote/", "name": "reddit_let_them_eat_cake"},
        {"url": "https://www.history.com/news/did-marie-antoinette-really-say-let-them-eat-cake", "name": "history_marie_antoinette"},
        {"url": "https://encyclopedia.ushmm.org/content/en/article/adolf-hitler-early-years-1889-1921", "name": "ushmm_hitler_1889_1921"},
        {"url": "https://en.wikipedia.org/wiki/Adolf_Hitler", "name": "wiki_adolf_hitler"},
        {"url": "https://encyclopedia.ushmm.org/content/en/article/adolf-hitler-early-years-1889-1913", "name": "ushmm_hitler_1889_1913"},
        {"url": "https://www.history.com/articles/adolf-hitler", "name": "history_adolf_hitler"},
        {"url": "https://www.bbc.co.uk/teach/articles/zbrx8xs", "name": "bbc_teach_zbrx8xs"}
    ]
    
    print(f"Initializing {len(approved_sites)} approved websites...")
    for site in approved_sites:
        # Skip if already in APPROVED_WEBSITES
        if site["name"] in APPROVED_WEBSITES:
            print(f"Website {site['name']} already exists, skipping")
            continue
            
        try:
            print(f"Processing website: {site['url']}")
            result = fetch_and_process_webpage(site["url"], site["name"])
            
            if result['success']:
                # Initialize a processor for this web source
                processor = RAGProcessor(result['index_path'], result['chunks_path'])
                WEB_SOURCE_PROCESSORS[site["name"]] = processor
                
                # Add to approved websites
                APPROVED_WEBSITES[site["name"]] = {
                    'url': site["url"],
                    'name': site["name"],
                    'description': f"Approved website: {site['url']}",
                    'path': result['path'],
                    'index_path': result['index_path'],
                    'chunks_path': result['chunks_path'],
                    'chunk_count': result['chunk_count'],
                    'added_on': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                print(f"Successfully added {site['name']} with {result['chunk_count']} chunks")
            else:
                print(f"Failed to process {site['url']}: {result.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"Error processing {site['url']}: {str(e)}")
    
    # Save websites to disk
    save_websites_to_disk()
    print(f"Finished initializing approved websites. Total: {len(APPROVED_WEBSITES)}")

# Load saved data on startup
load_sources_from_disk()
load_selected_sources_from_disk()
load_websites_from_disk()
verify_data_integrity()

def fetch_and_process_webpage(url, source_name):
    """Fetch content from a web page and process it for vector storage"""
    try:
        print(f"\n[WEB SCRAPING] Starting to process: {url} (name: {source_name})")
        
        # Send request to URL
        print(f"[WEB SCRAPING] Sending HTTP request to {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
        
        # Parse HTML content
        print(f"[WEB SCRAPING] Got response ({len(response.content)} bytes), parsing HTML")
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract text content (remove script and style elements)
        print("[WEB SCRAPING] Removing script, style, header, footer, nav elements")
        for script in soup(['script', 'style', 'header', 'footer', 'nav']):
            script.extract()
        
        # Get text content
        text = soup.get_text(separator=' ', strip=True)
        print(f"[WEB SCRAPING] Extracted {len(text)} characters of text content")
        
        # Create folder for this web source
        source_folder = os.path.join(WEB_DATA_FOLDER, source_name)
        if not os.path.exists(source_folder):
            os.makedirs(source_folder)
            print(f"[WEB SCRAPING] Created folder: {source_folder}")
        
        # Process text into chunks (similar to PDF processing but simplified)
        chunks = []
        # Simple chunking by paragraphs (you may want a more sophisticated approach)
        paragraphs = [p for p in text.split('\n') if p.strip()]
        print(f"[WEB SCRAPING] Split content into {len(paragraphs)} paragraphs")
        
        for i, paragraph in enumerate(paragraphs):
            if paragraph.strip():
                chunks.append({
                    "chunk_id": f"web_{source_name}_{i}",
                    "text": paragraph,
                    "page": "web",
                    "section": source_name,
                    "source_url": url
                })
        
        print(f"[WEB SCRAPING] Created {len(chunks)} chunks from paragraphs")
        
        # Save chunks to JSON
        chunks_path = os.path.join(source_folder, 'chunks.json')
        with open(chunks_path, 'w') as f:
            json.dump(chunks, f)
        print(f"[WEB SCRAPING] Saved chunks to {chunks_path}")
        
        # Build vector store
        index_path = os.path.join(source_folder, 'faiss_index.bin')
        processed_chunks_path = os.path.join(source_folder, 'processed_chunks.json')
        
        # Create vector store
        print(f"[WEB SCRAPING] Building vector store from chunks")
        vector_store = build_vector_store(chunks_path, index_path, processed_chunks_path)
        print(f"[WEB SCRAPING] Vector store built and saved to {index_path}")
        
        print(f"[WEB SCRAPING] Successfully processed {url}")
        return {
            'success': True,
            'path': source_folder,
            'index_path': index_path,
            'chunks_path': processed_chunks_path,
            'chunk_count': len(chunks)
        }
    except Exception as e:
        print(f"[WEB SCRAPING] Error processing web page {url}: {str(e)}")
        return {'success': False, 'error': str(e)}

initialize_processors()

# Check if we need to initialize approved websites
if not APPROVED_WEBSITES:
    initialize_approved_websites()

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
        
        # Check if this source has already been processed
        if source_name in SOURCES:
            # Source already exists, no need to process again
            return jsonify({
                'success': True, 
                'message': f'Document {source_name} already processed and available in the database',
                'sources': SOURCES
            })
        
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
            
            # Save sources to disk for persistence
            save_sources_to_disk()
            
            # If no sources were previously selected, select this one
            if not SELECTED_SOURCES:
                SELECTED_SOURCES.append(source_name)
                save_selected_sources_to_disk()
            
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
    
    print(f"Query processing started. Query: '{query}'")
    print(f"Initial selected sources from request: {selected_sources}")
    
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
        print("No sources available in the database")
        return jsonify({'success': False, 'error': 'No textbooks available in the database. Please upload a textbook first.'})
    
    if not selected_sources:
        # If no sources were explicitly provided in the request, use the globally selected sources
        selected_sources = SELECTED_SOURCES
        print(f"No sources in request, using global SELECTED_SOURCES: {selected_sources}")
        
        # If still no sources selected, default to all available sources
        if not selected_sources:
            selected_sources = list(SOURCES.keys())
            print(f"No global sources found, defaulting to all: {selected_sources}")
    
    print(f"Final selected sources for query: {selected_sources}")
    
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
                    print(f"Created new processor for source: {source}")
                processors.append(SOURCE_PROCESSORS[source])
                print(f"Added processor for source: {source}")
            else:
                print(f"Warning: Source '{source}' requested but not found in available SOURCES")
        
        print(f"Total processors loaded: {len(processors)}")
        
        if not processors:
            return jsonify({'success': False, 'error': 'No valid sources selected'})
        
        # Process the query using all selected processors
        combined_results = []
        all_sections = set()
        all_pages = set()
        
        # Query each processor separately
        for i, processor in enumerate(processors):
            source_name = selected_sources[i] if i < len(selected_sources) else "Unknown Source"
            print(f"Processing query with source: {source_name}")
            query_id = str(uuid.uuid4())
            
            # Process query with processor
            result = processor.process_query(
                query_id,
                query,
                top_k=3  # Fewer results per source when using multiple sources
            )
            
            print(f"Result from {source_name}: Answer length: {len(result['Answer'])}, Sections: {result['Sections']}, Pages: {result['Pages']}")
            
            # Only add non-empty results
            if result['Answer'].strip():
                # Collect sections with source attribution
                if result['Sections']:
                    sections = result['Sections'].split(', ')
                    for section in sections:
                        if section:
                            all_sections.add(f"{source_name}: {section}")
                
                # Collect pages with source attribution
                if result['Pages']:
                    pages = result['Pages'].split(', ')
                    all_pages.add(f"{source_name}: {', '.join(pages)}")
                
                # Add to combined results
                combined_results.append({
                    'source': source_name,
                    'answer': result['Answer']
                })
                print(f"Added result from {source_name} to combined results")
        
        print(f"Total results collected: {len(combined_results)}")
        
        if not combined_results:
            return jsonify({
                'success': True,
                'answer': "I couldn't find relevant information about this in the selected textbooks.",
                'sections': [],
                'pages': [],
                'language': 'en'
            })
        
        # Combine answers from all sources
        combined_answer = ""
        
        for i, result in enumerate(combined_results):
            if i > 0:
                combined_answer += "\n\n"
            combined_answer += f"From {result['source']}:\n{result['answer']}"
        
        print(f"Final combined answer length: {len(combined_answer)}, Sections: {len(all_sections)}, Pages: {len(all_pages)}")
        
        # Store in query history for CSV export
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        history_entry = {
            'timestamp': timestamp,
            'query_id': str(uuid.uuid4()),
            'query': query,
            'answer': combined_answer,
            'sources': ", ".join(selected_sources),
            'sections': ", ".join(all_sections),
            'pages': ", ".join(all_pages),
            'language': 'en'
        }
        QUERY_HISTORY.append(history_entry)
        
        return jsonify({
            'success': True,
            'answer': combined_answer,
            'sections': list(all_sections),
            'pages': list(all_pages),
            'language': 'en'
        })
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        import traceback
        traceback.print_exc()
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
    """Return list of available sources with selection status"""
    global SELECTED_SOURCES
    
    # If no sources are selected but we have sources, select all by default
    if not SELECTED_SOURCES and SOURCES:
        SELECTED_SOURCES = list(SOURCES.keys())
    
    # Create a dictionary with source information and selection status
    sources_with_status = {}
    for source_name, source_data in SOURCES.items():
        # Copy the source data
        source_info = source_data.copy()
        # Add selection status
        source_info['selected'] = source_name in SELECTED_SOURCES
        sources_with_status[source_name] = source_info
    
    return jsonify(sources_with_status)

@app.route('/sources/<source>', methods=['DELETE'])
def remove_source(source):
    """Remove a source"""
    global SELECTED_SOURCES
    
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
            
            # Remove from selected sources if it's there
            if source in SELECTED_SOURCES:
                SELECTED_SOURCES.remove(source)
                save_selected_sources_to_disk()
            
            # Save changes to disk
            save_sources_to_disk()
            
            return jsonify({'success': True, 'sources': SOURCES})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    else:
        return jsonify({'success': False, 'error': 'Source not found'})

@app.route('/reset-all', methods=['POST'])
def reset_all_sources():
    """Reset all sources and clear vector stores"""
    global SOURCES, SOURCE_PROCESSORS, QUERY_HISTORY, SELECTED_SOURCES
    
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
        SELECTED_SOURCES = []
        
        # Clear persistent data
        if os.path.exists(SOURCES_FILE):
            os.remove(SOURCES_FILE)
        if os.path.exists(SELECTED_SOURCES_FILE):
            os.remove(SELECTED_SOURCES_FILE)
        
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
        
        # Save to disk for persistence
        save_websites_to_disk()
        
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
            
            # Save changes to disk
            save_websites_to_disk()
            
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
    
    print(f"\n[WEB SEARCH] Query: '{query}'")
    print(f"[WEB SEARCH] Selected websites: {selected_websites}")
    
    if not query:
        return jsonify({'success': False, 'error': 'No query provided'})
    
    if not APPROVED_WEBSITES:
        print("[WEB SEARCH] No approved websites available")
        return jsonify({'success': False, 'error': 'No approved websites available'})
    
    if not selected_websites:
        # If no websites selected, use all available
        selected_websites = list(APPROVED_WEBSITES.keys())
        print(f"[WEB SEARCH] No specific websites selected, using all: {selected_websites}")
    
    try:
        # Load processors for selected websites
        processors = []
        for website in selected_websites:
            if website in APPROVED_WEBSITES:
                if website not in WEB_SOURCE_PROCESSORS:
                    # Create processor for this website
                    website_data = APPROVED_WEBSITES[website]
                    print(f"[WEB SEARCH] Creating new processor for {website}")
                    processor = RAGProcessor(website_data['index_path'], website_data['chunks_path'])
                    WEB_SOURCE_PROCESSORS[website] = processor
                processors.append(WEB_SOURCE_PROCESSORS[website])
                print(f"[WEB SEARCH] Added processor for {website}")
            else:
                print(f"[WEB SEARCH] WARNING: Website {website} not found in approved websites")
        
        if not processors:
            print("[WEB SEARCH] No valid website processors available")
            return jsonify({'success': False, 'error': 'No valid websites selected'})
        
        print(f"[WEB SEARCH] Using {len(processors)} website processors")
        
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
            
            print(f"[WEB SEARCH] Processing query with website: {website_name}")
            
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
                print(f"[WEB SEARCH] Got valid answer from {website_name}")
            else:
                print(f"[WEB SEARCH] No valid answer from {website_name}")
        
        if not combined_results:
            print("[WEB SEARCH] No results found from any web source")
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
            
            combined_answer += f"From {source_name}:\n{result['Answer']}"
            
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
        
        print(f"[WEB SEARCH] Combined answer from {len(combined_results)} sources")
        print(f"[WEB SEARCH] Web sources: {', '.join([s['name'] for s in all_sources])}")
        
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
        print(f"[WEB SEARCH] Error processing web query: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/combined-query', methods=['POST'])
def process_combined_query():
    """Process a query using both textbooks and approved web sources"""
    global QUERY_HISTORY
    
    data = request.json
    query = data.get('query')
    selected_sources = data.get('sources', [])
    selected_websites = data.get('websites', [])
    language = data.get('language', 'en')
    
    print(f"\n[COMBINED SEARCH] Query: '{query}'")
    print(f"[COMBINED SEARCH] Selected textbooks: {selected_sources}")
    print(f"[COMBINED SEARCH] Selected websites: {selected_websites}")
    print(f"[COMBINED SEARCH] Language: {language}")
    
    if not query:
        return jsonify({'success': False, 'error': 'No query provided'})
    
    # Check if we have any sources available
    if not selected_sources and not selected_websites:
        print("[COMBINED SEARCH] No sources selected (neither textbooks nor websites)")
        return jsonify({'success': False, 'error': 'No sources selected'})
    
    try:
        # Create a custom request object for textbook query
        textbook_data = {}
        if selected_sources and SOURCES:
            textbook_data = {
                'query': query,
                'sources': selected_sources,
                'language': data.get('language', 'en')
            }
            print(f"[COMBINED SEARCH] Will query textbooks: {selected_sources}")
        
        # Create a custom request object for web query
        web_data = {}
        if selected_websites and APPROVED_WEBSITES:
            web_data = {
                'query': query,
                'websites': selected_websites,
                'language': data.get('language', 'en')
            }
            print(f"[COMBINED SEARCH] Will query websites: {selected_websites}")
        
        # Process with textbook sources
        textbook_result = None
        if textbook_data:
            print("[COMBINED SEARCH] Processing textbook sources...")
            with app.test_request_context(
                '/query', 
                method='POST',
                json=textbook_data
            ):
                textbook_result = process_query()
                if hasattr(textbook_result, 'get_json'):
                    textbook_result = textbook_result.get_json()
            
            print(f"[COMBINED SEARCH] Textbook search successful: {textbook_result.get('success', False)}")
        
        # Process with web sources
        web_result = None
        if web_data:
            print("[COMBINED SEARCH] Processing web sources...")
            with app.test_request_context(
                '/web-query', 
                method='POST',
                json=web_data
            ):
                web_result = process_web_query()
                if hasattr(web_result, 'get_json'):
                    web_result = web_result.get_json()
            
            print(f"[COMBINED SEARCH] Web search successful: {web_result.get('success', False)}")
        
        # Combine results
        combined_answer = ""
        all_sections = []
        all_pages = []
        all_web_sources = []
        
        # Add textbook results
        if textbook_result and textbook_result.get('success'):
            print("[COMBINED SEARCH] Adding textbook results to answer")
            combined_answer += "From textbook sources:\n" + textbook_result['answer']
            all_sections.extend(textbook_result.get('sections', []))
            all_pages.extend(textbook_result.get('pages', []))
        
        # Add web results
        if web_result and web_result.get('success'):
            print("[COMBINED SEARCH] Adding web results to answer")
            if combined_answer:
                combined_answer += "\n\nFrom web sources:\n"
            else:
                combined_answer += "From web sources:\n"
            combined_answer += web_result['answer']
            all_sections.extend(web_result.get('sections', []))
            all_pages.extend(web_result.get('pages', []))
            all_web_sources = web_result.get('web_sources', [])
            
            print(f"[COMBINED SEARCH] Web sources added: {len(all_web_sources)}")
        
        # If no results were found
        if not combined_answer:
            print("[COMBINED SEARCH] No answers found from any source")
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
            'language': language,
            'type': 'combined_query'
        }
        QUERY_HISTORY.append(history_entry)
        
        print(f"[COMBINED SEARCH] Returning combined answer with {len(all_sections)} sections, {len(all_pages)} pages, {len(all_web_sources)} web sources")
        
        return jsonify({
            'success': True,
            'answer': combined_answer,
            'sections': all_sections,
            'pages': all_pages,
            'language': language,
            'web_sources': all_web_sources
        })
    except Exception as e:
        print(f"[COMBINED SEARCH] Error processing combined query: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/books', methods=['GET'])
def get_books():
    """Get a list of all processed books/sources in the system with metadata"""
    global SELECTED_SOURCES
    book_list = []
    
    print(f"Processing get_books request. SOURCES: {SOURCES}")
    
    # If no sources are selected but we have sources, select all by default
    if not SELECTED_SOURCES and SOURCES:
        SELECTED_SOURCES = list(SOURCES.keys())
        print(f"No selected sources, defaulting to all: {SELECTED_SOURCES}")
    
    # Process the SOURCES dictionary into a more user-friendly format
    for source_name, source_data in SOURCES.items():
        try:
            chunks_path = source_data['chunks_path']
            processed_date = time.ctime(os.path.getctime(chunks_path)) if os.path.exists(chunks_path) else 'Unknown'
            
            book_info = {
                'name': source_name,
                'chunk_count': source_data['chunk_count'],
                'processed_date': processed_date,
                'selected': source_name in SELECTED_SOURCES
            }
            book_list.append(book_info)
            print(f"Added book: {book_info}")
        except Exception as e:
            print(f"Error processing book {source_name}: {str(e)}")
    
    # Sort books by name
    book_list.sort(key=lambda x: x['name'])
    
    response = {
        'success': True,
        'count': len(book_list),
        'books': book_list
    }
    print(f"Returning response: {response}")
    return jsonify(response)

@app.route('/books-page')
def books_page():
    """Render the books page"""
    return render_template('books.html')

@app.route('/selected-sources', methods=['GET'])
def get_selected_sources():
    """Get the user's selected sources for queries"""
    global SELECTED_SOURCES
    
    # If no sources are selected but we have sources available, select all by default
    if not SELECTED_SOURCES and SOURCES:
        SELECTED_SOURCES = list(SOURCES.keys())
    
    return jsonify({
        'success': True,
        'selected_sources': SELECTED_SOURCES
    })

@app.route('/save-sources', methods=['POST'])
def save_selected_sources():
    """Save the user's selected sources for queries"""
    global SELECTED_SOURCES
    
    data = request.json
    selected_sources = data.get('selected_sources', [])
    
    # Validate that all selected sources exist
    for source in selected_sources:
        if source not in SOURCES:
            return jsonify({
                'success': False,
                'error': f'Source "{source}" does not exist'
            })
    
    # Save the selected sources
    SELECTED_SOURCES = selected_sources
    
    # Save to disk for persistence
    success = save_selected_sources_to_disk()
    
    return jsonify({
        'success': True,
        'message': 'Selected sources saved successfully',
        'selected_sources': SELECTED_SOURCES,
        'persistent': success
    })

@app.route('/check-books', methods=['GET'])
def check_books():
    """Check if there are any books already in the system"""
    return jsonify({
        'success': True,
        'has_books': len(SOURCES) > 0,
        'book_count': len(SOURCES),
        'book_names': list(SOURCES.keys())
    })

@app.route('/initialize-websites', methods=['POST'])
def initialize_all_websites():
    """Initialize all approved websites at once."""
    try:
        # If no websites defined yet, initialize with the pre-approved list
        if not APPROVED_WEBSITES:
            app.logger.info("Initializing approved websites list")
            # Call the function that has the correct list already defined
            initialize_approved_websites()
            return jsonify({"success": True, "message": "All approved websites initialized successfully"})
        else:
            # If websites already exist, just return success
            return jsonify({"success": True, "message": f"Websites already initialized. Total: {len(APPROVED_WEBSITES)}"})
    except Exception as e:
        app.logger.error(f"Error initializing websites: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/websites-page')
def websites_page():
    """Render the websites management page"""
    return render_template('websites.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)