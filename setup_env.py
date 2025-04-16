#!/usr/bin/env python3
"""
Environment Setup Script

This script helps set up the necessary environment variables for the RAG system.
It creates a .env file from user input or from .env.example.
"""

import os
import shutil
import sys

from utils import print_header, print_success, print_error, print_info, print_warning

def setup_env():
    """Set up environment variables for the RAG system"""
    print_header("ENVIRONMENT SETUP")
    
    # Check if .env already exists
    if os.path.exists(".env"):
        print_warning("A .env file already exists!")
        overwrite = input("Do you want to overwrite it? (y/n): ").strip().lower()
        if overwrite != 'y':
            print_info("Setup cancelled. Your existing .env file was not modified.")
            return
    
    # Check if .env.example exists
    if os.path.exists(".env.example"):
        print_info("Found .env.example template")
        use_template = input("Use the template as a base? (y/n): ").strip().lower()
        
        if use_template == 'y':
            # Copy the template
            shutil.copy(".env.example", ".env")
            print_success(".env file created from template")
        else:
            # Create a new .env file
            with open(".env", "w") as f:
                f.write("# Gemini API Key from Google AI Studio\n")
                f.write("GEMINI_API_KEY=\n")
            print_success("Empty .env file created")
    else:
        # No template found, create a new .env file
        with open(".env", "w") as f:
            f.write("# Gemini API Key from Google AI Studio\n")
            f.write("GEMINI_API_KEY=\n")
        print_success("Empty .env file created")
    
    # Ask for API key
    print_info("You'll need a Google Gemini API key to use this application")
    print_info("Get one from: https://aistudio.google.com/app/apikey")
    
    api_key = input("Enter your Gemini API key (leave blank to enter later): ").strip()
    
    if api_key:
        # Update the API key in the .env file
        with open(".env", "r") as f:
            lines = f.readlines()
        
        with open(".env", "w") as f:
            for line in lines:
                if line.startswith("GEMINI_API_KEY="):
                    f.write(f"GEMINI_API_KEY={api_key}\n")
                else:
                    f.write(line)
        
        print_success("API key saved to .env file")
    else:
        print_info("No API key provided. You can edit the .env file manually later.")
    
    print_info("Environment setup complete! You can now run the application.")

if __name__ == "__main__":
    try:
        setup_env()
    except KeyboardInterrupt:
        print_warning("\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Error during setup: {str(e)}")
        sys.exit(1) 