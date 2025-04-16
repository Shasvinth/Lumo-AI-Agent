"""
Utility functions for the RAG chatbot system.
This file contains helper functions used throughout the application.
"""

import os
import sys
from datetime import datetime

# Terminal colors for prettier output
class Colors:
    """ANSI color codes for terminal output styling"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    """Print a styled header in the terminal"""
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(80)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")

def print_step(step_number, step_name):
    """Print a styled step header in the terminal"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}[Step {step_number}] {step_name}{Colors.ENDC}")
    print(f"{Colors.BLUE}{'-' * 60}{Colors.ENDC}")

def print_success(text):
    """Print a success message"""
    print(f"{Colors.GREEN}{Colors.BOLD}✓ {text}{Colors.ENDC}")

def print_error(text):
    """Print an error message"""
    print(f"{Colors.RED}{Colors.BOLD}✗ Error: {text}{Colors.ENDC}")

def print_warning(text):
    """Print a warning message"""
    print(f"{Colors.YELLOW}⚠ Warning: {text}{Colors.ENDC}")

def print_info(text):
    """Print an info message"""
    print(f"{Colors.CYAN}ℹ {text}{Colors.ENDC}")

def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    """
    Print a progress bar in the terminal.
    
    Args:
        iteration: Current iteration
        total: Total iterations
        prefix: Prefix string
        suffix: Suffix string
        length: Bar length
        fill: Bar fill character
    """
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    
    sys.stdout.write(f'\r{prefix} |{Colors.BLUE}{bar}{Colors.ENDC}| {percent}% {suffix}')
    
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

def get_timestamp():
    """Get current timestamp in readable format"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def validate_file_exists(file_path):
    """Validate if a file exists, raise error if it doesn't"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return True

def limit_text_for_display(text, max_length=100):
    """Limit text length for display purposes"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."

def format_metadata(sections, pages):
    """Format metadata for display"""
    if not sections:
        sections_str = "N/A"
    else:
        sections_str = "; ".join(sections)
        
    if not pages:
        pages_str = "N/A"
    else:
        pages_str = ", ".join(map(str, pages))
        
    return sections_str, pages_str 