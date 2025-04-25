import re
import sys

try:
    with open('app.py', 'r') as f:
        content = f.read()
    
    # Try to compile the file
    compile(content, 'app.py', 'exec')
    print("No syntax errors in the full file")
    
except SyntaxError as e:
    print(f"Syntax error found at line {e.lineno}, column {e.offset}: {e.msg}")
    
    # Read the file line by line
    with open('app.py', 'r') as f:
        lines = f.readlines()
    
    # Display context around the error
    error_line = e.lineno
    start_line = max(0, error_line - 5)
    end_line = min(len(lines), error_line + 5)
    
    print("\nContext around the error:")
    for i in range(start_line, end_line):
        prefix = ">>> " if i+1 == error_line else "    "
        print(f"{prefix}{i+1}: {lines[i].rstrip()}")
        
    print("\nChecking for unbalanced brackets/parentheses near the error...")
    # Extract a chunk of code around the error to check for unbalanced brackets
    chunk_start = max(0, error_line - 20)
    chunk_end = min(len(lines), error_line + 20)
    chunk = ''.join(lines[chunk_start:chunk_end])
    
    # Count brackets
    open_parens = chunk.count('(')
    close_parens = chunk.count(')')
    open_brackets = chunk.count('[')
    close_brackets = chunk.count(']')
    open_braces = chunk.count('{')
    close_braces = chunk.count('}')
    
    print(f"Parentheses: {open_parens} opening, {close_parens} closing")
    print(f"Brackets: {open_brackets} opening, {close_brackets} closing")
    print(f"Braces: {open_braces} opening, {close_braces} closing")
    
    if open_parens != close_parens or open_brackets != close_brackets or open_braces != close_braces:
        print("Unbalanced brackets detected in the chunk surrounding the error!") 