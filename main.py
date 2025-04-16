"""
Main RAG Pipeline Module

This module integrates all components of the RAG (Retrieval-Augmented Generation) system:
1. PDF processing
2. Vector store building
3. Query processing

This script serves as the main entry point for running the complete pipeline.

Usage example:
    python main.py --pdf textbook.pdf --queries queries.json --output-csv results.csv
"""

import os
import argparse
import gc  # For garbage collection
import time  # For better error recovery
import sys
from dotenv import load_dotenv

from pdf_processor import process_pdf
from embedding_store import build_vector_store
from rag_processor import RAGProcessor
from utils import (
    print_header, print_step, print_success, print_error, 
    print_warning, print_info, validate_file_exists
)

def main():
    """
    Run the full RAG pipeline:
    1. Process PDF into chunks
    2. Build vector store
    3. Process queries and generate answers
    """
    # Print welcome header
    print_header("RAG CHATBOT FOR HISTORY TEXTBOOK")
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run the full RAG pipeline")
    
    # PDF processing args
    parser.add_argument("--pdf", help="Path to the textbook PDF file")
    parser.add_argument("--chunks-output", default="data/output/chunks.json", 
                        help="Output path for chunks JSON")
    parser.add_argument("--chunk-size", type=int, default=500, 
                        help="Size of each chunk in characters")
    parser.add_argument("--overlap", type=float, default=0.2, 
                        help="Overlap between chunks (as a fraction)")
    
    # Vector store args
    parser.add_argument("--index-output", default="data/output/faiss_index.bin", 
                        help="Output path for FAISS index")
    parser.add_argument("--processed-chunks", default="data/output/processed_chunks.json", 
                        help="Output path for processed chunks")
    
    # Query processing args
    parser.add_argument("--queries", help="Path to the queries JSON file")
    parser.add_argument("--output-csv", default="data/output/queries_output.csv", 
                        help="Output CSV path")
    parser.add_argument("--top-k", type=int, default=5, 
                        help="Number of contexts to retrieve per query")
    
    # Pipeline control
    parser.add_argument("--skip-pdf", action="store_true", 
                        help="Skip PDF processing step")
    parser.add_argument("--skip-vectorstore", action="store_true", 
                        help="Skip vector store building step")
    
    # Memory management
    parser.add_argument("--memory-efficient", action="store_true",
                       help="Enable memory efficient mode (slower but uses less RAM)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(args.chunks_output), exist_ok=True)
    os.makedirs(os.path.dirname(args.index_output), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print_error("GEMINI_API_KEY not found in environment variables")
        print_info("Please create a .env file with your Gemini API key (GEMINI_API_KEY=your_key)")
        sys.exit(1)
        
    # Memory efficient mode
    if args.memory_efficient:
        print_info("Memory efficient mode enabled (will run slower but use less RAM)")
        
    # Step 1: Process PDF
    if not args.skip_pdf:
        if not args.pdf:
            print_error("PDF path is required unless --skip-pdf is used")
            sys.exit(1)
            
        try:
            # Validate PDF file exists
            validate_file_exists(args.pdf)
            
            # Process PDF
            process_pdf(args.pdf, args.chunks_output, args.chunk_size, args.overlap)
            
            # Force garbage collection
            gc.collect()
            
            # Small delay to ensure all resources are released
            time.sleep(1)
        except Exception as e:
            print_error(f"PDF processing failed: {str(e)}")
            sys.exit(1)
    else:
        print_info("Skipping PDF processing step...")
        
    # Step 2: Build vector store
    if not args.skip_vectorstore:
        try:
            chunks_path = args.chunks_output if not args.skip_pdf else args.processed_chunks
            
            # Validate chunks file exists
            validate_file_exists(chunks_path)
            
            # Build vector store
            build_vector_store(
                chunks_path, 
                args.index_output, 
                args.processed_chunks,
                memory_efficient=args.memory_efficient
            )
            
            # Force garbage collection
            gc.collect()
            
            # Small delay to ensure all resources are released
            time.sleep(1)
        except Exception as e:
            print_error(f"Vector store building failed: {str(e)}")
            sys.exit(1)
    else:
        print_info("Skipping vector store building step...")
        
    # Step 3: Process queries
    if args.queries:
        try:
            # Validate query file exists
            validate_file_exists(args.queries)
            
            # Validate index and chunks files exist
            validate_file_exists(args.index_output)
            validate_file_exists(args.processed_chunks)
            
            # Create processor
            processor = RAGProcessor(
                args.index_output, 
                args.processed_chunks
            )
            
            # Process queries
            processor.process_queries_file(args.queries, args.output_csv, args.top_k)
            
            # Final result
            print_success(f"Results saved to {args.output_csv}")
        except Exception as e:
            print_error(f"Query processing failed: {str(e)}")
            sys.exit(1)
    else:
        print_info("No queries file provided, skipping query processing step.")
        
    # Pipeline completed
    print_header("RAG PIPELINE COMPLETED SUCCESSFULLY")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_warning("\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        sys.exit(1) 