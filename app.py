"""
Streamlit Web Interface

This module provides an interactive web interface for the RAG system.
It allows users to:
1. Upload a PDF textbook and query JSON file
2. Process the PDF and build a vector store
3. Test individual queries or process batches of queries
4. Download results

Usage:
    streamlit run app.py
"""

import streamlit as st
import os
import json
import gc  # For garbage collection
import time  # For better response
import sys
from dotenv import load_dotenv

from pdf_processor import process_pdf
from embedding_store import build_vector_store, EmbeddingStore
from rag_processor import RAGProcessor
from utils import (
    limit_text_for_display, format_metadata, print_info, 
    print_success, print_error, print_warning
)

# Load environment variables
load_dotenv()

# Set page title and configuration
st.set_page_config(
    page_title="History Unlocked - RAG Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-top: 0;
    }
    .success-message {
        color: #2E7D32;
        font-weight: bold;
    }
    .error-message {
        color: #C62828;
        font-weight: bold;
    }
    .info-message {
        color: #0277BD;
    }
    .warning-message {
        color: #EF6C00;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<p class="main-header">📚 History Unlocked</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">RAG Chatbot for Grade 11 History Textbook</p>', unsafe_allow_html=True)

# Sidebar for configurations
st.sidebar.title("Configuration")

# API Key input
api_key = st.sidebar.text_input(
    "Gemini API Key", 
    value=os.getenv("GEMINI_API_KEY", ""), 
    type="password",
    help="Enter your Gemini API key from Google AI Studio"
)
if api_key:
    os.environ["GEMINI_API_KEY"] = api_key
    
# Check if API key is provided
if not api_key:
    st.sidebar.warning("⚠️ Please enter your Gemini API Key to use this application.")

# File upload section
st.sidebar.header("Upload Files")

# PDF upload
pdf_file = st.sidebar.file_uploader(
    "Upload Textbook PDF", 
    type="pdf",
    help="Upload your Grade 11 History textbook in PDF format"
)
if pdf_file:
    try:
        with open("uploaded_textbook.pdf", "wb") as f:
            f.write(pdf_file.getbuffer())
        st.sidebar.success("✅ PDF uploaded successfully!")
    except Exception as e:
        st.sidebar.error(f"❌ Error saving PDF: {str(e)}")

# Queries upload
queries_file = st.sidebar.file_uploader(
    "Upload Queries JSON (optional)", 
    type="json",
    help="Upload a JSON file with queries in the required format"
)
if queries_file:
    try:
        queries = json.loads(queries_file.getvalue())
        with open("uploaded_queries.json", "w") as f:
            json.dump(queries, f)
        st.sidebar.success(f"✅ Loaded {len(queries)} queries!")
    except Exception as e:
        st.sidebar.error(f"❌ Error processing queries: {str(e)}")

# Parameters
st.sidebar.header("Parameters")
chunk_size = st.sidebar.slider(
    "Chunk Size", 
    200, 1000, 500,
    help="Size of each text chunk in characters. Larger chunks provide more context but may be less focused"
)
overlap = st.sidebar.slider(
    "Overlap Ratio", 
    0.0, 0.5, 0.2,
    help="Overlap between consecutive chunks to maintain context across chunk boundaries"
)
top_k = st.sidebar.slider(
    "Number of chunks to retrieve", 
    1, 10, 5,
    help="Number of most relevant chunks to retrieve for each query"
)

# Memory efficiency
memory_efficient = st.sidebar.checkbox(
    "Memory Efficient Mode",
    help="Enable for lower memory usage (may be slower but helps prevent crashes)"
)

# About section
with st.sidebar.expander("About This App"):
    st.write("""
    This application uses Retrieval-Augmented Generation (RAG) to answer questions from a Grade 11 History textbook.
    
    **How it works:**
    1. Upload your textbook PDF
    2. Process the PDF into chunks
    3. Build a vector store for semantic search
    4. Ask questions and get answers with references
    
    **Technologies used:** Python, FAISS, Google Gemini 1.5 Flash, Streamlit
    """)

# Processing section
st.header("Processing Pipeline")

# Initialize session state
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'index_built' not in st.session_state:
    st.session_state.index_built = False
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False

# Step 1: Process PDF
col1, col2 = st.columns([3, 1])
with col1:
    st.subheader("Step 1: Process PDF")
    st.write("Split the textbook into chunks and extract metadata (sections and page numbers)")
with col2:
    if st.button("Process PDF", disabled=not pdf_file):
        with st.spinner("Processing PDF..."):
            try:
                # Force garbage collection before starting
                gc.collect()
                
                # Process PDF with parameters
                chunks = process_pdf("uploaded_textbook.pdf", "chunks.json", chunk_size, overlap)
                st.session_state.pdf_processed = True
                
                # Clear memory
                gc.collect()
                
                # Success message
                st.success(f"✅ Processed {len(chunks)} chunks from the PDF")
            except Exception as e:
                st.error(f"❌ Error processing PDF: {str(e)}")

# Step 2: Build Vector Store
col1, col2 = st.columns([3, 1])
with col1:
    st.subheader("Step 2: Build Vector Store")
    st.write("Create embeddings and build a search index for semantic retrieval")
with col2:
    if st.button("Build Index", disabled=not st.session_state.pdf_processed):
        with st.spinner("Building vector index..."):
            try:
                # Force garbage collection before starting
                gc.collect()
                
                # Build vector store
                build_vector_store("chunks.json", "faiss_index.bin", "processed_chunks.json")
                st.session_state.index_built = True
                
                # Initialize the processor
                st.session_state.processor = RAGProcessor(
                    "faiss_index.bin", 
                    "processed_chunks.json"
                )
                
                # Clear memory
                gc.collect()
                
                # Success message
                st.success("✅ Vector index built successfully!")
            except Exception as e:
                st.error(f"❌ Error building vector store: {str(e)}")

# Step 3: Query Testing
st.header("Query Testing")

# Single query input
query = st.text_input(
    "Enter a history question:",
    placeholder="e.g., What were the causes of the Uva Rebellion?"
)

if query and st.button("Submit Query"):
    if not st.session_state.processor:
        st.error("⚠️ Please build the vector index first")
    else:
        with st.spinner("Generating answer..."):
            try:
                # Process query
                result = st.session_state.processor.process_query("test", query, top_k)
                
                # Display answer
                st.subheader("Answer")
                st.write(result["Answer"])
                
                # Show metadata
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Sections:** {result['Sections']}")
                with col2:
                    st.write(f"**Pages:** {result['Pages']}")
                
                # Expandable source text
                with st.expander("View Source Text"):
                    st.write(result["Context"])
                
                # Clear memory
                gc.collect()
                
            except Exception as e:
                st.error(f"❌ Error processing query: {str(e)}")

# Batch processing of queries
if queries_file and st.session_state.processor:
    st.header("Batch Query Processing")
    if st.button("Process All Queries"):
        with st.spinner("Processing all queries..."):
            try:
                # Process all queries
                results = st.session_state.processor.process_queries_file(
                    "uploaded_queries.json", "queries_output.csv", top_k
                )
                st.success(f"✅ Processed {len(results)} queries and saved to queries_output.csv")
                
                # Clear memory
                gc.collect()
                
                # Display results preview
                st.subheader("Results Preview")
                # Show first 3 results
                for i, result in enumerate(results[:3]):
                    with st.expander(f"Query {i+1}: {limit_text_for_display(result['Answer'], 50)}"):
                        st.write(f"**Question:** {queries[i]['query']}")
                        st.write(f"**Answer:** {result['Answer']}")
                        st.write(f"**Sections:** {result['Sections']}")
                        st.write(f"**Pages:** {result['Pages']}")
                
                # Download link
                with open("queries_output.csv", "rb") as file:
                    st.download_button(
                        label="Download Results CSV",
                        data=file,
                        file_name="queries_output.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"❌ Error processing queries: {str(e)}")

# Footer
st.markdown("---")
st.markdown("**History Unlocked** - RAG Chatbot for Grade 11 History Textbook | Powered by Google Gemini 1.5 Flash")

# Add debug expander if needed
with st.expander("Debug Information", expanded=False):
    st.write(f"PDF Processed: {st.session_state.pdf_processed}")
    st.write(f"Index Built: {st.session_state.index_built}")
    if 'processor' in st.session_state and st.session_state.processor:
        st.write("RAG Processor: Initialized")
    else:
        st.write("RAG Processor: Not initialized") 