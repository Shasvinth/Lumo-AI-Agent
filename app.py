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
from components.utils.utils import limit_text_for_display

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

# Add folders to app configuration
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DATA_FOLDER'] = DATA_FOLDER
app.config['WEB_DATA_FOLDER'] = WEB_DATA_FOLDER
app.config['EXPORTS_FOLDER'] = EXPORTS_FOLDER

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
    """Initialize processors for all sources and websites"""
    global SOURCES, SOURCE_PROCESSORS, APPROVED_WEBSITES, WEB_SOURCE_PROCESSORS
    
    print("Initializing processors...")
    
    # Initialize source processors
    if SELECTED_SOURCES:
        print(f"Initializing {len(SELECTED_SOURCES)} selected source processors...")
        count = 0
        for source in SELECTED_SOURCES:
            if source in SOURCES:
                try:
                    source_data = SOURCES[source]
                    processor = RAGProcessor(source_data['index_path'], source_data['chunks_path'])
                    SOURCE_PROCESSORS[source] = processor
                    count += 1
                    print(f"Initialized processor for source: {source} ({count}/{len(SELECTED_SOURCES)})")
                except Exception as e:
                    print(f"Failed to initialize processor for source {source}: {str(e)}")
        print(f"Initialized {count} source processors")
    
    # Initialize website processors
    if APPROVED_WEBSITES:
        print(f"Initializing {len(APPROVED_WEBSITES)} website processors...")
        count = 0
        total = len(APPROVED_WEBSITES)
        for name, website in APPROVED_WEBSITES.items():
            try:
                processor = RAGProcessor(website['index_path'], website['chunks_path'])
                WEB_SOURCE_PROCESSORS[name] = processor
                count += 1
                print(f"Initialized processor for website: {name} ({count}/{total})")
            except Exception as e:
                print(f"Failed to initialize processor for website {name}: {str(e)}")
        print(f"Initialized {count} website processors")
        
    print("Processor initialization complete.")

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
                    'page_title': result.get('page_title', site["name"]),
                    'descriptive_title': result.get('descriptive_title', ''),
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

# Initialize processor dictionaries in app.config
try:
    print("Initializing app.config['processors'] dictionary")
    app.config['processors'] = {}
    
    # First try to copy existing processors
    if SOURCE_PROCESSORS:
        print(f"Using {len(SOURCE_PROCESSORS)} existing processors from SOURCE_PROCESSORS")
        app.config['processors'] = SOURCE_PROCESSORS
    
    # Make sure we can access any already processed sources
    if SOURCES and not app.config['processors']:
        print("No existing processors, but sources are available. Will initialize on demand.")
        
    print(f"Processor initialization complete. {len(app.config['processors'])} processors registered.")
except Exception as e:
    print(f"ERROR during processor initialization: {str(e)}")
    import traceback
    traceback.print_exc()
    # Create empty processors dict as fallback
    app.config['processors'] = {}

# Initialize processors for sources
initialize_processors()

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
        
        # Extract page title
        page_title = soup.title.string if soup.title else source_name
        page_title = page_title.strip()
        print(f"[WEB SCRAPING] Extracted page title: {page_title}")
        
        # Create a more descriptive title
        meta_description = soup.find('meta', attrs={'name': 'description'})
        if meta_description and meta_description.get('content'):
            descriptive_title = f"{page_title} - {meta_description.get('content')}"
        else:
            # If no meta description, use first paragraph or heading as additional description
            first_p = soup.find('p')
            first_h = soup.find(['h1', 'h2'])
            additional_text = ""
            if first_h and first_h.text.strip():
                additional_text = first_h.text.strip()
            elif first_p and first_p.text.strip():
                additional_text = first_p.text.strip()
                
            if additional_text:
                descriptive_title = f"{page_title} - {additional_text[:100]}"
            else:
                descriptive_title = page_title
        
        print(f"[WEB SCRAPING] Created descriptive title: {descriptive_title}")
        
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
                    "source_url": url,
                    "title": page_title,
                    "descriptive_title": descriptive_title
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
            'chunk_count': len(chunks),
            'page_title': page_title,
            'descriptive_title': descriptive_title
        }
    except Exception as e:
        print(f"[WEB SCRAPING] Error processing web page {url}: {str(e)}")
        return {'success': False, 'error': str(e)}

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

def format_ai_response(response):
    """
    Format AI response to remove redundant statements about lack of information
    
    Args:
        response (str): The original AI response
        
    Returns:
        str: Formatted response without redundant statements
    """
    # List of repetitive phrases to detect
    repetitive_phrases = [
        "I do not have enough information to answer accurately",
        "The provided text focuses on",
        "Therefore, I do not have enough information",
        "I don't have enough information",
        "I do not have sufficient information",
        "The text does not contain information about",
        "does not contain information about"
    ]
    
    # If response contains any of these phrases multiple times, simplify it
    count = 0
    for phrase in repetitive_phrases:
        if phrase.lower() in response.lower():
            count += 1
    
    # If we have multiple occurrences of these phrases, simplify the response
    if count > 1:
        lines = response.split('\n')
        simplified_lines = []
        seen_phrases = set()
        
        # Keep track of non-redundant content
        has_useful_content = False
        
        for line in lines:
            # Check if this line contains any repetitive phrase
            contains_repetitive = False
            for phrase in repetitive_phrases:
                if phrase.lower() in line.lower():
                    contains_repetitive = True
                    # If we've seen this kind of phrase before, skip this line
                    for seen in seen_phrases:
                        if (phrase.lower() in seen.lower()) or (seen.lower() in phrase.lower()):
                            break
                    else:
                        # This is the first time we're seeing this phrase
                        seen_phrases.add(line)
                        simplified_lines.append(line)
                    break
            
            # If this line doesn't contain repetitive phrases, keep it
            if not contains_repetitive:
                simplified_lines.append(line)
                if line.strip() and not line.startswith("Based on") and not line.startswith("I apologize"):
                    has_useful_content = True
        
        # If we have useful content mixed with repetitive phrases, only remove duplicates
        if has_useful_content:
            return '\n'.join(simplified_lines)
        else:
            # If it's all about lack of information, simplify drastically
            return "I don't have enough information in the provided sources to answer this question accurately."
    
    # If not repetitive, return the original response
    return response

@app.route('/query', methods=['POST'])
def process_query():
    """Process a query against the selected sources using the RAG processor"""
    global QUERY_HISTORY
    
    data = request.json
    query_text = data.get('query', '')
    selected_sources = data.get('sources', [])
    use_markdown = data.get('useMarkdown', False)
    requested_language = data.get('language', 'en')
    
    print(f"Processing query: {query_text}")
    print(f"Selected sources: {selected_sources}")
    
    try:
        # Fix for selected_sources being passed as boolean instead of list
        if isinstance(selected_sources, bool):
            print(f"Converting selected_sources from boolean {selected_sources} to list")
            # If True, use all available sources from SELECTED_SOURCES
            if selected_sources:
                selected_sources = list(SELECTED_SOURCES)
                print(f"Using all selected sources: {selected_sources}")
            else:
                selected_sources = []
                print("No sources selected (boolean False)")
        
        # If no sources specified, use all selected sources
        if not selected_sources and SELECTED_SOURCES:
            selected_sources = list(SELECTED_SOURCES)
            print(f"No specific sources requested, using all selected: {selected_sources}")
        
        # Validate query
        if not query_text:
            return jsonify({'answer': 'No query provided', 'sources': [], 'pages': []}), 200
        
        # Validate sources
        if not selected_sources:
            error_msg = "No sources selected. Please select at least one source."
                
            # Translate error message if needed
            if requested_language != 'en':
                error_msg = translate_to_language(error_msg, requested_language)
                    
            return jsonify({
                'answer': error_msg,
                'sources': [],
                'pages': []
            }), 200
            
        # Initialize all processors if not already done
        if 'processors' not in app.config:
            print("WARNING: processors not found in app.config, initializing now")
            app.config['processors'] = {}
        
        # Combined list to store all results
        all_results = []
        all_sections = []
        all_pages = []
        
        # Log the query
        print(f"Processing query: {query_text}")
        print(f"Selected sources: {selected_sources}")
        
        # Process each selected source
        for source_name in selected_sources:
            print(f"Processing source: {source_name}")
            
            # Check if the source has a processor
            if source_name not in app.config['processors']:
                print(f"Processor not found for {source_name}, attempting to create one")
                # Find the source file and paths based on existing SOURCES data
                try:
                    if source_name in SOURCES:
                        source_data = SOURCES[source_name]
                        index_path = source_data['index_path']
                        chunks_path = source_data['chunks_path']
                        
                        # Check if files exist
                        if not os.path.exists(index_path):
                            print(f"ERROR: Index file not found: {index_path}")
                            continue
                            
                        if not os.path.exists(chunks_path):
                            print(f"ERROR: Chunks file not found: {chunks_path}")
                            continue
                        
                        # Create processor
                        processor = RAGProcessor(index_path=index_path, chunks_path=chunks_path)
                        app.config['processors'][source_name] = processor
                        SOURCE_PROCESSORS[source_name] = processor
                        print(f"Created processor for: {source_name}")
                    else:
                        print(f"ERROR: Source {source_name} not found in SOURCES dictionary")
                        continue
                except Exception as e:
                    print(f"ERROR initializing processor for {source_name}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Get processor for this source
            try:
                processor = app.config['processors'][source_name]
                if processor is None:
                    print(f"ERROR: Processor for {source_name} is None")
                    continue
                    
                # Process the query
                try:
                    print(f"Sending query to processor for {source_name}")
            result = processor.process_query(
                        query_id=str(uuid.uuid4()), 
                        query_text=query_text,
                        language=requested_language
                    )
                    
                    # DEBUG: Print full result structure
                    print(f"[DEBUG] Full result from {source_name}:")
                    print(f"Result type: {type(result)}")
                    print(f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
                    for key, value in result.items():
                        print(f"  - {key}: {type(value)} = {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")
                
                # Add to combined results
                    all_results.append({
                    'source': source_name,
                        'answer': result.get('answer', '')  # Use get() for safety
                    })
                    
                    # Add source-specific sections and pages with verbose error checking
                    print(f"[DEBUG] Processing sections and pages")
                    if 'sections' in result:
                        print(f"[DEBUG] Sections found: {type(result['sections'])} = {result['sections']}")
                        if isinstance(result['sections'], list):
                            for section in result['sections']:
                                try:
                                    section = str(section).strip()
                                    if section and section not in all_sections:
                                        all_sections.append(section)
                                except Exception as section_error:
                                    print(f"[DEBUG] Error processing section {section}: {str(section_error)}")
                        else:
                            print(f"[DEBUG] Sections not a list: {type(result['sections'])}")
                    else:
                        print("[DEBUG] No sections found in result")
                    
                    if 'pages' in result:
                        print(f"[DEBUG] Pages found: {type(result['pages'])} = {result['pages']}")
                        if isinstance(result['pages'], list):
                            for page in result['pages']:
                                try:
                                    page = str(page).strip()
                                    if page and page not in all_pages:
                                        all_pages.append(page)
                                except Exception as page_error:
                                    print(f"[DEBUG] Error processing page {page}: {str(page_error)}")
                        else:
                            print(f"[DEBUG] Pages not a list: {type(result['pages'])}")
                    else:
                        print("[DEBUG] No pages found in result")
                    
                    print(f"Processed query against source: {source_name}")
                except Exception as e:
                    print(f"ERROR processing query against {source_name}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
            except Exception as e:
                print(f"ERROR accessing processor for {source_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        # Combine all answers into one
        if not all_results:
            # No results from any source
            error_msg = "Failed to process query against any source. Please try again or select different sources."
            
            # Translate error message if needed
            if requested_language != 'en':
                error_msg = translate_to_language(error_msg, requested_language)
                
            return jsonify({
                'answer': error_msg,
                'sources': [],
                'pages': []
            }), 200
        
        # If we have only one result, use it directly
        if len(all_results) == 1:
            combined_answer = all_results[0]['answer']
        else:
            # Combine answers with source attribution
            combined_answer = "Here are answers from different sources:\n\n"
            for result in all_results:
                combined_answer += f"From {result['source']}:\n{result['answer']}\n\n"
        
        # Format the response to remove redundancy
        formatted_answer = format_ai_response(combined_answer)
        
        # Save query to history
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        query_record = {
            'timestamp': timestamp,
            'query': query_text,
            'answer': formatted_answer,  # Use the formatted answer
            'sources': ', '.join(selected_sources),
            'sections': ', '.join(all_sections),
            'pages': ', '.join(all_pages)
        }
        
        # Load existing history
        history_path = os.path.join(app.config['UPLOAD_FOLDER'], 'history.json')
        history = []
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    history = json.load(f)
            except:
                history = []
        
        # Add new record and save
        history.append(query_record)
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        # Return combined results
        return jsonify({
            'success': True,
            'answer': formatted_answer,  # Use the formatted answer
            'sources': all_sections,
            'pages': all_pages
        }), 200
        
    except Exception as e:
        print(f"Error in process_query: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Get detailed error information for debugging
        import sys
        exc_type, exc_value, exc_traceback = sys.exc_info()
        error_details = traceback.format_exception(exc_type, exc_value, exc_traceback)
        error_msg = f"Error processing query: {str(e)}\n\nDetails: {''.join(error_details[-5:])}"
        print(f"DETAILED ERROR: {error_msg}")
        
        # Provide a cleaner message to the user
        user_error_msg = f"Error processing query: {str(e)}"
        
        # Translate error message if needed
        if 'requested_language' in locals() and requested_language != 'en':
            user_error_msg = translate_to_language(user_error_msg, requested_language)
            
        return jsonify({
            'answer': user_error_msg,
            'sources': [],
            'pages': []
        }), 500

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
            'page_title': result.get('page_title', name),
            'descriptive_title': result.get('descriptive_title', ''),
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
    language = data.get('language', 'en')  # Get user language preference
    
    # Add more detailed debugging
    print(f"\n[WEB SEARCH] Query: '{query}'")
    print(f"[WEB SEARCH] Selected websites: {selected_websites}")
    print(f"[WEB SEARCH] Total approved websites: {len(APPROVED_WEBSITES)}")
    print(f"[WEB SEARCH] Requested language: {language}")
    
    if not query:
        return jsonify({'success': False, 'error': 'No query provided'})
    
    if not APPROVED_WEBSITES:
        print("[WEB SEARCH] No approved websites available")
        error_msg = 'No approved websites available'
        if language != 'en':
            error_msg = translate_to_language(error_msg, language)
        return jsonify({'success': False, 'error': error_msg})
    
    if not selected_websites:
        # If no websites selected, use all available
        selected_websites = list(APPROVED_WEBSITES.keys())
        print(f"[WEB SEARCH] No specific websites selected, using all: {selected_websites}")
    
    try:
        # Generate embedding for the query to use for semantic title matching
        import google.generativeai as genai
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        # Set up the embedding model (reuse from RAGProcessor if possible)
        embedding_model = None
        for key, processor in WEB_SOURCE_PROCESSORS.items():
            if processor and hasattr(processor, 'embedding_model'):
                embedding_model = processor.embedding_model
                break
        
        if not embedding_model:
            # If no processor has an embedding model, create one
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                embedding_model = genai.GenerativeModel("embedding-001")
        
        # Function to get title relevance score
        def get_title_relevance(website_name, query_text):
            website = APPROVED_WEBSITES.get(website_name)
            if not website:
                return 0
                
            # Get the descriptive title
            title = website.get('descriptive_title', website.get('page_title', ''))
            if not title:
                return 0
                
            # First check for PDF files which may have generic titles but specific content
            is_pdf = False
            url = website.get('url', '').lower()
            if url.endswith('.pdf') or '/pdf/' in url:
                print(f"[WEB SEARCH] {website_name} is a PDF document, giving higher base relevance")
                is_pdf = True
                # Give PDF files a base relevance score
                base_relevance = 0.3
            else:
                base_relevance = 0
            
            # Simple keyword matching for efficiency
            query_words = set(query_text.lower().split())
            title_words = set(title.lower().split())
            
            # Count matching words
            matching_words = query_words.intersection(title_words)
            word_match_score = len(matching_words) / len(query_words) if query_words else 0
            
            # Combine scores - PDF files get a boost plus any word matching
            if is_pdf:
                return max(base_relevance, word_match_score)
            else:
                return word_match_score
            
        # Filter websites by title relevance
        filtered_websites = []
        relevance_scores = {}
        
        print(f"[WEB SEARCH] Filtering {len(selected_websites)} websites by title relevance")
        for website_name in selected_websites:
            relevance = get_title_relevance(website_name, query)
            relevance_scores[website_name] = relevance
            
            # Keep websites with any relevance
            if relevance > 0:
                filtered_websites.append(website_name)
                print(f"[WEB SEARCH] Website '{website_name}' has title relevance score: {relevance:.2f}")
            else:
                print(f"[WEB SEARCH] Website '{website_name}' has no title relevance")
        
        # If no relevant websites found, use a subset of all websites to avoid empty results
        if not filtered_websites:
            print("[WEB SEARCH] No websites with relevant titles found, using subset of all websites")
            filtered_websites = selected_websites[:min(3, len(selected_websites))]
            print(f"[WEB SEARCH] Using {len(filtered_websites)} websites as fallback")
        else:
            # Sort by relevance
            filtered_websites.sort(key=lambda x: relevance_scores.get(x, 0), reverse=True)
            # Limit to top 5 most relevant
            filtered_websites = filtered_websites[:min(5, len(filtered_websites))]
            print(f"[WEB SEARCH] Selected top {len(filtered_websites)} websites by title relevance")
        
        # Print info about the filtered websites
        for website_name in filtered_websites:
            website = APPROVED_WEBSITES.get(website_name, {})
            title = website.get('descriptive_title', website.get('page_title', ''))
            print(f"[WEB SEARCH] Will search website: {website_name} - {title}")
            print(f"[WEB SEARCH] Relevance score: {relevance_scores.get(website_name, 0):.2f}")
        
        # Print detailed info about website data for the filtered websites
        for website_name in filtered_websites:
            if website_name in APPROVED_WEBSITES:
                website_data = APPROVED_WEBSITES[website_name]
                print(f"[WEB SEARCH] Website {website_name}:")
                print(f"  - URL: {website_data.get('url', 'N/A')}")
                print(f"  - Title: {website_data.get('page_title', 'N/A')}")
                print(f"  - Descriptive Title: {website_data.get('descriptive_title', 'N/A')}")
                print(f"  - Chunks path: {website_data.get('chunks_path', 'N/A')}")
                print(f"  - Index path: {website_data.get('index_path', 'N/A')}")
                print(f"  - Chunk count: {website_data.get('chunk_count', 0)}")
                
                # Check if files exist
                if os.path.exists(website_data.get('chunks_path', '')):
                    print(f"  - Chunks file exists and is {os.path.getsize(website_data.get('chunks_path', ''))} bytes")
                else:
                    print(f"  - WARNING: Chunks file does not exist")
                    
                if os.path.exists(website_data.get('index_path', '')):
                    print(f"  - Index file exists and is {os.path.getsize(website_data.get('index_path', ''))} bytes")
                else:
                    print(f"  - WARNING: Index file does not exist")
        
        # Load processors for selected websites
        processors = []
        website_names = []
        source_urls = {}
        
        for website in filtered_websites:
            if website in APPROVED_WEBSITES:
                if website not in WEB_SOURCE_PROCESSORS:
                    # Create processor for this website
                    website_data = APPROVED_WEBSITES[website]
                    print(f"[WEB SEARCH] Creating new processor for {website}")
                    processor = RAGProcessor(website_data['index_path'], website_data['chunks_path'])
                    WEB_SOURCE_PROCESSORS[website] = processor
                processors.append(WEB_SOURCE_PROCESSORS[website])
                website_names.append(website)
                # Store URL for later reference
                source_urls[website] = APPROVED_WEBSITES[website]['url']
                print(f"[WEB SEARCH] Added processor for {website}")
            else:
                print(f"[WEB SEARCH] WARNING: Website {website} not found in approved websites")
        
        if not processors:
            print("[WEB SEARCH] No valid website processors available")
            error_msg = 'No valid websites selected'
            if language != 'en':
                error_msg = translate_to_language(error_msg, language)
            return jsonify({'success': False, 'error': error_msg})
        
        print(f"[WEB SEARCH] Starting sequential search with {len(processors)} website processors")
        
        # Process query using all selected web sources sequentially to avoid threading issues
        import time
        import gc  # For garbage collection
        
        start_time = time.time()
        combined_results = []
        used_sources = set()
        
        # Process each website sequentially instead of in parallel
        for idx, processor in enumerate(processors):
            website_name = website_names[idx]
            query_id = str(uuid.uuid4())
            
            site_start_time = time.time()
            print(f"[WEB SEARCH] Processing {website_name}...")
            
            try:
                # Process query (always in English for now, translation happens later)
            result = processor.process_query(
                query_id,
                query,
                    top_k=8  # Increase from 5 to 8 to get more context
                )
                
                site_end_time = time.time()
                processing_time = site_end_time - site_start_time
                print(f"[WEB SEARCH] {website_name} completed in {processing_time:.2f}s")
                
                # Add more detailed debug about the result
                answer_length = len(result['answer']) if 'answer' in result else 0
                context_length = len(result['context']) if 'context' in result else 0
                print(f"[WEB SEARCH] {website_name} result:")
                print(f"  - Answer length: {answer_length} chars")
                print(f"  - Context length: {context_length} chars")
                print(f"  - Has answer: {'Yes' if answer_length > 0 else 'No'}")
                
                # Lower threshold for accepting answers - accept anything with content
                if result['answer'] and len(result['answer'].strip()) > 0:
                combined_results.append(result)
                    used_sources.add(website_name)
                    print(f"[WEB SEARCH] Got valid answer from {website_name} ({processing_time:.2f}s)")
                    print(f"[WEB SEARCH] Answer preview: {result['answer'][:100]}...")
            else:
                    print(f"[WEB SEARCH] No valid answer from {website_name} ({processing_time:.2f}s)")
                    # Print the context to debug why no answer was generated
                    if 'context' in result and result['context']:
                        print(f"[WEB SEARCH] Context preview: {result['context'][:100]}...")
                    else:
                        print(f"[WEB SEARCH] No context available")
                
                # Force garbage collection after each website
                gc.collect()
                
            except Exception as e:
                print(f"[WEB SEARCH] Error processing website {website_name}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        end_time = time.time()
        total_time = end_time - start_time
        print(f"[WEB SEARCH] Sequential processing completed in {total_time:.2f} seconds")
        
        if not combined_results:
            print("[WEB SEARCH] No results found from any web source")
            not_found_msg = "I couldn't find relevant information from the approved web sources."
            if language != 'en':
                not_found_msg = translate_to_language(not_found_msg, language)
            return jsonify({
                'success': True,
                'answer': not_found_msg,
                'sections': [],
                'pages': [],
                'language': language,
                'web_sources': [],
                'processing_time': f"{total_time:.2f}s"
            })
        
        # Format strings for translation
        source_prefix = "From"
        
        # If not English, translate the source prefix
        if language != 'en':
            source_prefix = translate_to_language(source_prefix, language)
        
        # Combine results (simple approach - concatenate with source attribution)
        combined_answer = ""
        english_answer = ""  # Store original English answer for history
        all_sections = set()
        all_sources = []
        
        for i, result in enumerate(combined_results):
            source_name = list(used_sources)[i] if i < len(used_sources) else "Unknown"
            source_url = source_urls.get(source_name, "")
            
            # Add source attribution
            if i > 0:
                combined_answer += "\n\n"
                english_answer += "\n\n"
            
            # Store original English answer
            english_answer += f"From {source_name}:\n{result['answer']}"
            
            # Translate answer if needed
            answer = result['answer']
            if language != 'en':
                answer = translate_to_language(answer, language)
            
            combined_answer += f"{source_prefix} {source_name}:\n{answer}"
            
            # Collect sections
            if 'sections' in result and isinstance(result['sections'], list):
                # Handle sections as a list
                for section in result['sections']:
                    section = str(section).strip()
                    if section and section not in all_sections:
                        all_sections.append(section)
            
            if 'pages' in result and isinstance(result['pages'], list):
                # Handle pages as a list
                for page in result['pages']:
                    page = str(page).strip()
                    if page and page not in all_pages:
                        all_pages.append(page)
            
            # Add source reference
            all_sources.append({
                'name': source_name,
                'url': source_url
            })
        
        # Format page references for web sources (they're actually source names)
        page_refs = [f"Source: {s['name']}" for s in all_sources]
        
        print(f"[WEB SEARCH] Combined answer from {len(combined_results)} sources in {total_time:.2f}s")
        print(f"[WEB SEARCH] Web sources: {', '.join([s['name'] for s in all_sources])}")
        print(f"[WEB SEARCH] Combined answer length: {len(combined_answer)} chars")
        
        # Store in query history (original English answer)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        history_entry = {
            'timestamp': timestamp,
            'query_id': str(uuid.uuid4()),
            'query': query,
            'answer': english_answer, # Store original English answer
            'sources': ", ".join([s['name'] for s in all_sources]),
            'sections': ", ".join(all_sections),
            'pages': ", ".join(page_refs),
            'language': language,
            'type': 'web_query',
            'processing_time': f"{total_time:.2f}s"
        }
        QUERY_HISTORY.append(history_entry)
        
        return jsonify({
            'success': True,
            'answer': combined_answer,
            'sections': list(all_sections),
            'pages': page_refs,
            'language': language,
            'web_sources': all_sources,
            'processing_time': f"{total_time:.2f}s"
        })
    except Exception as e:
        print(f"[WEB SEARCH] Error processing web query: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Get detailed error information for debugging
        import sys
        exc_type, exc_value, exc_traceback = sys.exc_info()
        error_details = traceback.format_exception(exc_type, exc_value, exc_traceback)
        error_msg = f"Error processing web query: {str(e)}\n\nDetails: {''.join(error_details[-5:])}"
        print(f"[WEB SEARCH] DETAILED ERROR: {error_msg}")
        
        # Provide a cleaner message to the user
        user_error_msg = str(e)
        
        if language != 'en':
            user_error_msg = translate_to_language(user_error_msg, language)
        return jsonify({'success': False, 'error': user_error_msg})

        return ""
        return text

@app.route('/combined-query', methods=['POST'])
def process_combined_query():
    """Process a query using both textbooks and approved web sources"""
    global QUERY_HISTORY
    
    data = request.json
    query = data.get('query', '')
    selected_sources = data.get('sources', [])
    selected_websites = data.get('websites', [])
    use_markdown = data.get('useMarkdown', False)
    language = data.get('language', 'en')  # Get the requested language
    
    print(f"\n[COMBINED SEARCH] Query: '{limit_text_for_display(query)}'")
    print(f"[COMBINED SEARCH] Selected textbooks: {selected_sources}")
    print(f"[COMBINED SEARCH] Selected websites: {selected_websites}")
    print(f"[COMBINED SEARCH] Use markdown: {use_markdown}")
    print(f"[COMBINED SEARCH] Requested language: {language}")
    
    if not query:
        return jsonify({'success': False, 'error': 'No query provided'})
    
    # Check if we have any sources available
    if not selected_sources and not selected_websites:
        print("[COMBINED SEARCH] No sources selected (neither textbooks nor websites)")
        error_msg = 'No sources selected'
        if language != 'en':
            error_msg = translate_to_language(error_msg, language)
        return jsonify({'success': False, 'error': error_msg})
    
    try:
        # Filter websites by title relevance first
        if selected_websites and APPROVED_WEBSITES:
            # Function to get title relevance score - same as in process_web_query
            def get_title_relevance(website_name, query_text):
                website = APPROVED_WEBSITES.get(website_name)
                if not website:
                    return 0
                    
                # Get the descriptive title
                title = website.get('descriptive_title', website.get('page_title', ''))
                if not title:
                    return 0
                
                # First check for PDF files which may have generic titles but specific content
                is_pdf = False
                url = website.get('url', '').lower()
                if url.endswith('.pdf') or '/pdf/' in url:
                    print(f"[COMBINED SEARCH] {website_name} is a PDF document, giving higher base relevance")
                    is_pdf = True
                    # Give PDF files a base relevance score
                    base_relevance = 0.3
                else:
                    base_relevance = 0
                
                # Simple keyword matching for efficiency
                query_words = set(query_text.lower().split())
                title_words = set(title.lower().split())
                
                # Count matching words
                matching_words = query_words.intersection(title_words)
                word_match_score = len(matching_words) / len(query_words) if query_words else 0
                
                # Combine scores - PDF files get a boost plus any word matching
                if is_pdf:
                    return max(base_relevance, word_match_score)
                else:
                    return word_match_score
            
            # Filter websites by title relevance
            filtered_websites = []
            relevance_scores = {}
            
            print(f"[COMBINED SEARCH] Filtering {len(selected_websites)} websites by title relevance")
            for website_name in selected_websites:
                relevance = get_title_relevance(website_name, query)
                relevance_scores[website_name] = relevance
                
                # Keep websites with any relevance
                if relevance > 0:
                    filtered_websites.append(website_name)
                    print(f"[COMBINED SEARCH] Website '{website_name}' has title relevance score: {relevance:.2f}")
                else:
                    print(f"[COMBINED SEARCH] Website '{website_name}' has no title relevance")
            
            # If no relevant websites found, use a subset of all websites to avoid empty results
            if not filtered_websites:
                print("[COMBINED SEARCH] No websites with relevant titles found, using subset of all websites")
                filtered_websites = selected_websites[:min(3, len(selected_websites))]
                print(f"[COMBINED SEARCH] Using {len(filtered_websites)} websites as fallback")
            else:
                # Sort by relevance
                filtered_websites.sort(key=lambda x: relevance_scores.get(x, 0), reverse=True)
                # Limit to top 5 most relevant
                filtered_websites = filtered_websites[:min(5, len(filtered_websites))]
                print(f"[COMBINED SEARCH] Selected top {len(filtered_websites)} websites by title relevance")
                
            # Replace original selected_websites with filtered ones
            selected_websites = filtered_websites
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time
        import gc  # Garbage collector

        start_time = time.time()
        results = {}
        
        # Create a custom request object for textbook query
        textbook_data = {}
        if selected_sources and SOURCES:
            textbook_data = {
                'query': query,
                'sources': selected_sources,
                'useMarkdown': use_markdown,
                'language': 'en'  # Always process in English
            }
            print(f"[COMBINED SEARCH] Will query textbooks: {selected_sources}")
        
        # Create a custom request object for web query
        web_data = {}
        if selected_websites and APPROVED_WEBSITES:
            web_data = {
                'query': query,
                'websites': selected_websites,
                'useMarkdown': use_markdown,
                'language': 'en'  # Always process in English
            }
            print(f"[COMBINED SEARCH] Will query websites: {selected_websites}")
        
        # Function to process textbook query
        def process_textbook_query():
            print("[COMBINED SEARCH] Processing textbook sources...")
            with app.test_request_context(
                '/query', 
                method='POST',
                json=textbook_data
            ):
                result = process_query()
                if hasattr(result, 'get_json'):
                    return result.get_json()
                return result
        
        # Function to process web query - renamed to avoid recursion
        def process_web_search():
            print("[COMBINED SEARCH] Processing web sources...")
            with app.test_request_context(
                '/web-query', 
                method='POST',
                json=web_data
            ):
                result = process_web_query()
                if hasattr(result, 'get_json'):
                    return result.get_json()
                return result
        
        # Process searches sequentially to avoid OpenMP threading issues
        if textbook_data:
            print("[COMBINED SEARCH] Running textbook search")
            try:
                # Make sure processors dictionary is initialized
                if 'processors' not in app.config:
                    print("[COMBINED SEARCH] WARNING: processors not found in app.config, initializing now")
                    app.config['processors'] = SOURCE_PROCESSORS

                textbook_result = process_textbook_query()
                if isinstance(textbook_result, tuple):
                    # Handle tuple response (response, status_code)
                    results['textbook'] = textbook_result[0]
                    if hasattr(results['textbook'], 'get_json'):
                        results['textbook'] = results['textbook'].get_json()
                else:
                    results['textbook'] = textbook_result
                print("[COMBINED SEARCH] Textbook search completed")
                # Force garbage collection after task
                gc.collect()
            except Exception as e:
                print(f"[COMBINED SEARCH] Error processing textbook search: {str(e)}")
                import traceback
                traceback.print_exc()
                results['textbook'] = {'success': False, 'error': str(e)}
        
        if web_data:
            print("[COMBINED SEARCH] Running web search")
            try:
                web_result = process_web_search()
                if isinstance(web_result, tuple):
                    # Handle tuple response (response, status_code)
                    results['web'] = web_result[0]
                    if hasattr(results['web'], 'get_json'):
                        results['web'] = results['web'].get_json()
                else:
                    results['web'] = web_result
                print("[COMBINED SEARCH] Web search completed")
                # Force garbage collection after task
                gc.collect()
            except Exception as e:
                print(f"[COMBINED SEARCH] Error processing web search: {str(e)}")
                results['web'] = {'success': False, 'error': str(e)}
        
        end_time = time.time()
        print(f"[COMBINED SEARCH] Sequential processing completed in {end_time - start_time:.2f} seconds")
        
        # Check if we have any successful results
        textbook_result = results.get('textbook', {})
        web_result = results.get('web', {})
        
        textbook_success = textbook_result.get('success', False) if textbook_result else False
        web_success = web_result.get('success', False) if web_result else False
        
        if not textbook_success and not web_success:
            print("[COMBINED SEARCH] No successful results from any source")
            error_msg = 'No results found from any of the selected sources'
            if language != 'en':
                error_msg = translate_to_language(error_msg, language)
            return jsonify({
                'success': False, 
                'error': error_msg
            })
        
        # Format strings for translation
        textbook_header = "From textbooks:"
        
        # If not English, translate headers
        if language != 'en':
            textbook_header = translate_to_language(textbook_header, language)
            
        # Combine all successful results
        combined_answer = ""
        english_answer = ""  # For storing in history
        all_sections = []
        all_pages = []
        all_web_sources = []
        
        # Add textbook results
        if textbook_success:
            print("[COMBINED SEARCH] Adding textbook results")
            try:
                print(f"[DEBUG] Textbook result keys: {list(textbook_result.keys()) if isinstance(textbook_result, dict) else 'Not a dict'}")
                textbook_answer = textbook_result.get('answer', '')
                print(f"[DEBUG] Textbook answer type: {type(textbook_answer)}, length: {len(str(textbook_answer))}")
                
                if textbook_answer:
                    # Store original English answer
                    english_answer += "From textbooks:\n" + textbook_answer
                    
                    # Translate if needed
                    if language != 'en':
                        textbook_answer = translate_to_language(textbook_answer, language)
                        
                    combined_answer += textbook_header + "\n" + textbook_answer
                
                # Add sections and pages with detailed error checking
                print(f"[DEBUG] Processing textbook sections and pages")
                
                # Process sections with error checking
                if 'sections' in textbook_result:
                    print(f"[DEBUG] Textbook sections found: {type(textbook_result['sections'])} = {textbook_result['sections']}")
                    if isinstance(textbook_result['sections'], list):
                        for section in textbook_result['sections']:
                            try:
                                section_str = f"Textbook: {str(section)}"
                                if section_str not in all_sections:
                                    all_sections.append(section_str)
                            except Exception as section_error:
                                print(f"[DEBUG] Error processing textbook section {section}: {str(section_error)}")
                    else:
                        print(f"[DEBUG] Textbook sections not a list: {type(textbook_result['sections'])}")
                
                # Process pages with error checking
                if 'pages' in textbook_result:
                    print(f"[DEBUG] Textbook pages found: {type(textbook_result['pages'])} = {textbook_result['pages']}")
                    if isinstance(textbook_result['pages'], list):
                        for page in textbook_result['pages']:
                            try:
                                page_str = str(page)
                                if page_str not in all_pages:
                                    all_pages.append(page_str)
                            except Exception as page_error:
                                print(f"[DEBUG] Error processing textbook page {page}: {str(page_error)}")
                    else:
                        print(f"[DEBUG] Textbook pages not a list: {type(textbook_result['pages'])}")
            except Exception as textbook_error:
                print(f"[DEBUG] Error processing textbook results: {str(textbook_error)}")
                import traceback
                traceback.print_exc()
        
        # Add web results
        if web_success:
            print("[COMBINED SEARCH] Adding web results")
            try:
                print(f"[DEBUG] Web result keys: {list(web_result.keys()) if isinstance(web_result, dict) else 'Not a dict'}")
                web_answer = web_result.get('answer', '')
                print(f"[DEBUG] Web answer type: {type(web_answer)}, length: {len(str(web_answer))}")
                
                if web_answer:
            if combined_answer:
                        combined_answer += "\n\n"
                        english_answer += "\n\n"
                    
                    # Store original English answer
                    english_answer += web_answer
                    
                    # Translate if needed
                    if language != 'en':
                        web_answer = translate_to_language(web_answer, language)
                        
                    combined_answer += web_answer
                
                # Process sections with error checking
                print(f"[DEBUG] Processing web sections and pages")
                if 'sections' in web_result:
                    print(f"[DEBUG] Web sections found: {type(web_result['sections'])} = {web_result['sections']}")
                    if isinstance(web_result['sections'], list):
                        for section in web_result['sections']:
                            try:
                                section_str = str(section).strip()
                                if section_str and section_str not in all_sections:
                                    all_sections.append(section_str)
                            except Exception as section_error:
                                print(f"[DEBUG] Error processing web section {section}: {str(section_error)}")
            else:
                        print(f"[DEBUG] Web sections not a list: {type(web_result['sections'])}")
                
                # Process pages with error checking
                if 'pages' in web_result:
                    print(f"[DEBUG] Web pages found: {type(web_result['pages'])} = {web_result['pages']}")
                    if isinstance(web_result['pages'], list):
                        for page in web_result['pages']:
                            try:
                                page_str = str(page).strip()
                                if page_str and page_str not in all_pages:
                                    all_pages.append(page_str)
                            except Exception as page_error:
                                print(f"[DEBUG] Error processing web page {page}: {str(page_error)}")
                    else:
                        print(f"[DEBUG] Web pages not a list: {type(web_result['pages'])}")
                
                web_answer = web_result.get('answer', '')
                if 'web_sources' in web_result:
                    print(f"[DEBUG] Web sources found: {type(web_result['web_sources'])} = {web_result['web_sources']}")
                    try:
                        all_web_sources.extend(web_result['web_sources'])
                    except Exception as sources_error:
                        print(f"[DEBUG] Error processing web sources: {str(sources_error)}")
                        # Try to recover by adding each source one by one
                        if isinstance(web_result['web_sources'], list):
                            for source in web_result['web_sources']:
                                try:
                                    all_web_sources.append(source)
                                except Exception as single_source_error:
                                    print(f"[DEBUG] Error adding web source {source}: {str(single_source_error)}")
            except Exception as web_error:
                print(f"[DEBUG] Error processing web results: {str(web_error)}")
                import traceback
                traceback.print_exc()
        
        print(f"[COMBINED SEARCH] Combined answer created successfully")
        print(f"[COMBINED SEARCH] Final answer length: {len(combined_answer)}")
        print(f"[COMBINED SEARCH] Final sections: {len(all_sections)}")
        print(f"[COMBINED SEARCH] Final pages: {len(all_pages)}")
        print(f"[COMBINED SEARCH] Final web sources: {len(all_web_sources)}")
        
        # Store in query history (original English answer)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        history_entry = {
            'timestamp': timestamp,
            'query_id': str(uuid.uuid4()),
            'query': query,
            'answer': english_answer, # Store original English answer
            'sources': ", ".join([s['name'] for s in all_web_sources]) if all_web_sources else "Textbooks",
            'sections': ", ".join(all_sections),
            'pages': ", ".join(all_pages),
            'language': language,
            'type': 'combined_query'
        }
        QUERY_HISTORY.append(history_entry)
        
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
        import traceback
        traceback.print_exc()
        
        # Get detailed error information for debugging
        import sys
        exc_type, exc_value, exc_traceback = sys.exc_info()
        error_details = traceback.format_exception(exc_type, exc_value, exc_traceback)
        error_msg = f"Error processing combined query: {str(e)}\n\nDetails: {''.join(error_details[-5:])}"
        print(f"[COMBINED SEARCH] DETAILED ERROR: {error_msg}")
        
        # Provide a cleaner message to the user
        user_error_msg = f"Error processing combined query: {str(e)}"
        
        if language != 'en':
            user_error_msg = translate_to_language(user_error_msg, language)
        return jsonify({'success': False, 'error': user_error_msg})

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
    # Print clear message about where the app is running
    ip = '0.0.0.0'  # This makes it accessible from other machines
    port = 5001
    print("\n" + "="*60)
    print(f"LUMO AI Agent is running on:")
    print(f"Local URL:   http://127.0.0.1:{port}")
    print(f"Network URL: http://<your-ip-address>:{port}")
    print(f"Press CTRL+C to quit")
    print("="*60 + "\n")
    
    # Run the app on all network interfaces
    app.run(debug=True, host=ip, port=port)
