"""
PDF Processor Module

This module handles extracting text and metadata from PDF textbooks.
It splits the PDF into overlapping chunks and extracts section headings and page numbers.

Usage example:
    python pdf_processor.py textbook.pdf --output chunks.json
"""

import fitz  # PyMuPDF
import re
import json
import os
import gc  # For garbage collection
from tqdm import tqdm

# Import utility functions
from utils import (
    print_step, print_success, print_error, print_warning, 
    print_info, validate_file_exists, print_progress_bar
)

class PDFProcessor:
    """
    Process PDF files by extracting text and relevant metadata.
    
    This class handles PDF parsing, text extraction, and chunking with metadata
    like section titles and page numbers for later retrieval.
    """
    
    def __init__(self, pdf_path):
        """
        Initialize the PDF processor.
        
        Args:
            pdf_path (str): Path to the PDF file to process
        """
        # Validate that the PDF file exists
        validate_file_exists(pdf_path)
        
        # Initialize PDF path and document
        self.pdf_path = pdf_path
        print_info(f"Opening PDF: {os.path.basename(pdf_path)}")
        self.doc = fitz.open(pdf_path)
        self.num_pages = len(self.doc)
        print_info(f"Found {self.num_pages} pages in the document")
        
    def extract_text_with_metadata(self, chunk_size=500, overlap=0.2):
        """
        Extract text from PDF with chunk size and overlap.
        Detects section headings and tracks page numbers for each chunk.
        
        Args:
            chunk_size (int): Size of each chunk in characters
            overlap (float): Overlap between chunks as a fraction (0.0-1.0)
            
        Returns:
            list: List of dictionaries containing text chunks and metadata
        """
        # List to store all the chunks
        chunks = []
        
        # Set default section and calculate overlap size
        current_section = "Introduction"  # Default section if none found
        overlap_size = int(chunk_size * overlap)
        
        # Regular expression pattern to detect headings
        # This pattern looks for numbered sections (e.g., "1.2 The Great War")
        # or ALL CAPS section titles - adjust based on textbook format
        heading_pattern = re.compile(r'^(?:\d+\.)+\s+[A-Z][\w\s]+|^[A-Z][A-Z\s]+\b')
        
        # Variables to keep track of current chunk
        current_chunk = ""
        current_pages = set()  # Using a set to avoid duplicates
        
        # Process pages with progress bar
        print_info(f"Processing {self.num_pages} pages with chunk size {chunk_size} and {overlap*100}% overlap")
        for page_num in range(self.num_pages):
            # Update progress bar
            print_progress_bar(
                page_num + 1, 
                self.num_pages, 
                prefix=f'Processing pages:', 
                suffix=f'Page {page_num+1}/{self.num_pages}'
            )
            
            # Get page and extract text
            page = self.doc[page_num]
            text = page.get_text()
            
            # Check for section headings
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if heading_pattern.match(line):
                    # Found a new section heading
                    current_section = line.strip()
            
            # Add text to current chunk
            current_chunk += text
            current_pages.add(page_num + 1)  # 1-indexed page numbers
            
            # If chunk exceeds size, save it and start a new one with overlap
            if len(current_chunk) >= chunk_size:
                # Create a new chunk dictionary with metadata
                chunks.append({
                    "text": current_chunk,
                    "section": current_section,
                    "pages": sorted(list(current_pages))
                })
                
                # Start a new chunk with overlap
                words = current_chunk.split()
                overlap_text = " ".join(words[-overlap_size:]) if len(words) > overlap_size else ""
                current_chunk = overlap_text
                
                # Reset pages for new chunk, but keep current page
                current_pages = {page_num + 1}
                
                # Force garbage collection every 50 chunks to manage memory
                if len(chunks) % 50 == 0:
                    gc.collect()
        
        # Add the last chunk if not empty
        if current_chunk.strip():
            chunks.append({
                "text": current_chunk,
                "section": current_section,
                "pages": sorted(list(current_pages))
            })
            
        # Print completion message
        print_success(f"Created {len(chunks)} chunks from {self.num_pages} pages")
        return chunks
    
    def save_chunks(self, chunks, output_path='chunks.json'):
        """
        Save the extracted chunks to a JSON file
        
        Args:
            chunks (list): List of chunk dictionaries
            output_path (str): Path to save the JSON file
        """
        print_info(f"Saving {len(chunks)} chunks to {output_path}")
        try:
            with open(output_path, 'w') as f:
                json.dump(chunks, f, indent=2)
            print_success(f"Chunks saved successfully to {output_path}")
        except Exception as e:
            print_error(f"Failed to save chunks: {str(e)}")
            raise
            
    def close(self):
        """Close the PDF document to free resources"""
        self.doc.close()
        print_info("PDF document closed")
        # Force garbage collection
        gc.collect()

def process_pdf(pdf_path, output_path='chunks.json', chunk_size=500, overlap=0.2):
    """
    Process PDF and save chunks with metadata - main function
    
    Args:
        pdf_path (str): Path to the PDF file
        output_path (str): Path to save the JSON output
        chunk_size (int): Size of each chunk in characters
        overlap (float): Overlap between chunks as a fraction
        
    Returns:
        list: List of chunk dictionaries
    """
    print_step(1, "Processing PDF Document")
    processor = PDFProcessor(pdf_path)
    try:
        # Extract text and metadata
        chunks = processor.extract_text_with_metadata(chunk_size, overlap)
        
        # Save chunks to JSON
        processor.save_chunks(chunks, output_path)
        
        print_success(f"Successfully processed {len(chunks)} chunks from {processor.num_pages} pages")
        return chunks
    except Exception as e:
        print_error(f"Error processing PDF: {str(e)}")
        raise
    finally:
        # Always close the document to free resources
        processor.close()

if __name__ == "__main__":
    """Run as standalone script to process a PDF"""
    import argparse
    
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="Process PDF textbook into chunks with metadata")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--output", default="chunks.json", help="Output JSON file path")
    parser.add_argument("--chunk-size", type=int, default=500, help="Size of each chunk in characters")
    parser.add_argument("--overlap", type=float, default=0.2, help="Overlap between chunks (as a fraction)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process the PDF
    process_pdf(args.pdf_path, args.output, args.chunk_size, args.overlap) 