def check_file_for_syntax_errors(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # Try increasing ranges of lines
    for i in range(1, len(lines) + 1):
        try:
            code_snippet = ''.join(lines[:i])
            compile(code_snippet, filename, 'exec')
        except SyntaxError as e:
            print(f"Syntax error found at line {i}, line content:")
            print(f"Line {i}: {lines[i-1].strip()}")
            
            # Show context
            start = max(0, i-3)
            end = min(len(lines), i+2)
            print("\nContext:")
            for j in range(start, end):
                marker = ">>> " if j+1 == i else "    "
                print(f"{marker}Line {j+1}: {lines[j].strip()}")
            
            return
    
    print("No syntax errors found in the file.")

if __name__ == "__main__":
    check_file_for_syntax_errors('app.py') 