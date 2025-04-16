#!/usr/bin/env python3
"""
RAG Chatbot Runner Script

This script provides a simple command-line interface to run the RAG chatbot.
It offers options to start the web interface or run the pipeline directly.
"""

import os
import sys
import argparse
import subprocess
import webbrowser
import time

from utils import print_header, print_step, print_success, print_error, print_info, print_warning

def check_environment():
    """Check if the environment is set up correctly"""
    # Check if .env file exists
    if not os.path.exists(".env"):
        print_warning(".env file not found! Running setup...")
        try:
            subprocess.run([sys.executable, "setup_env.py"], check=True)
        except subprocess.CalledProcessError:
            print_error("Failed to set up environment")
            return False
    
    # Check if required directories exist
    os.makedirs("data/input", exist_ok=True)
    os.makedirs("data/output", exist_ok=True)
    
    return True

def run_streamlit():
    """Run the Streamlit web interface"""
    print_info("Starting Streamlit web interface...")
    print_info("Opening browser in 3 seconds...")
    
    # Start Streamlit in a separate process
    streamlit_process = subprocess.Popen(
        ["streamlit", "run", "app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait a moment to let Streamlit start
    time.sleep(3)
    
    # Open browser
    webbrowser.open("http://localhost:8501")
    
    print_info("Streamlit is running. Press Ctrl+C to stop.")
    
    try:
        # Keep the script running until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print_warning("\nStopping Streamlit...")
        streamlit_process.terminate()
        streamlit_process.wait()
        print_success("Streamlit stopped")

def run_pipeline(args):
    """Run the main RAG pipeline"""
    print_info("Running RAG pipeline...")
    
    # Build command
    cmd = [sys.executable, "main.py"]
    
    # Add arguments
    if args.pdf:
        cmd.extend(["--pdf", args.pdf])
    if args.queries:
        cmd.extend(["--queries", args.queries])
    if args.output:
        cmd.extend(["--output-csv", args.output])
    if args.top_k:
        cmd.extend(["--top-k", str(args.top_k)])
    if args.skip_pdf:
        cmd.append("--skip-pdf")
    if args.skip_vectorstore:
        cmd.append("--skip-vectorstore")
    if args.memory_efficient:
        cmd.append("--memory-efficient")
    
    # Run the command
    try:
        subprocess.run(cmd, check=True)
        print_success("Pipeline completed successfully")
    except subprocess.CalledProcessError as e:
        print_error(f"Pipeline failed with exit code {e.returncode}")
        return False
    
    return True

def main():
    """Main function to run the RAG chatbot"""
    print_header("HISTORY UNLOCKED - RAG CHATBOT")
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run the RAG chatbot")
    parser.add_argument("--web", action="store_true", help="Start the web interface (Streamlit app)")
    
    # Pipeline arguments
    parser.add_argument("--pdf", help="Path to the textbook PDF file")
    parser.add_argument("--queries", help="Path to the queries JSON file")
    parser.add_argument("--output", help="Path to the output CSV file")
    parser.add_argument("--top-k", type=int, help="Number of chunks to retrieve per query")
    parser.add_argument("--skip-pdf", action="store_true", help="Skip PDF processing step")
    parser.add_argument("--skip-vectorstore", action="store_true", help="Skip vector store building step")
    parser.add_argument("--memory-efficient", action="store_true",
                       help="Enable memory efficient mode (slower but uses less RAM)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Run the appropriate mode
    if args.web:
        run_streamlit()
    else:
        if args.pdf or args.queries or args.skip_pdf:
            success = run_pipeline(args)
            if not success:
                sys.exit(1)
        else:
            print_info("No actions specified. Use --web to start the web interface or provide pipeline arguments.")
            print_info("For help, run: python run.py --help")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_warning("\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        sys.exit(1) 