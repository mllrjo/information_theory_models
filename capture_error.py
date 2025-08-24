# File: capture_error.py
# Directory: ./

"""
Helper script to capture command output and parse errors automatically.
Usage: python capture_error.py <command>
"""

import subprocess
import sys
import os
from parse_errors import ErrorParser, format_error_summary


def capture_and_parse_command(command: list):
    """
    Run command, capture output, and parse any errors.
    
    Args:
        command: List of command parts (e.g., ['python', '-m', 'pytest', 'test.py'])
    """
    print(f"Running: {' '.join(command)}")
    print("-" * 50)
    
    try:
        # Run command and capture both stdout and stderr
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        # Print the original output first
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
            print()
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            print()
        
        # If there was an error, parse it
        if result.returncode != 0:
            print("\n" + "="*60)
            print("ERROR ANALYSIS")
            print("="*60)
            
            # Combine stdout and stderr for parsing
            full_output = result.stdout + "\n" + result.stderr
            
            parser = ErrorParser()
            summary = parser.parse_error(full_output)
            
            print(format_error_summary(summary))
            
            # Save error to file for future reference
            error_file = "last_error.txt"
            with open(error_file, 'w') as f:
                f.write("COMMAND: " + ' '.join(command) + "\n\n")
                f.write("STDOUT:\n" + result.stdout + "\n\n")
                f.write("STDERR:\n" + result.stderr + "\n\n")
                f.write("PARSED SUMMARY:\n" + format_error_summary(summary))
            
            print(f"\nFull error saved to: {error_file}")
            
        else:
            print("✓ Command completed successfully")
            
        return result.returncode
        
    except FileNotFoundError:
        print(f"Error: Command not found: {command[0]}")
        return 1
    except Exception as e:
        print(f"Error running command: {e}")
        return 1


def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) < 2:
        print("Usage: python capture_error.py <command> [args...]")
        print("\nExamples:")
        print("  python capture_error.py python -m pytest tests/test_utils/test_device_manager.py -v")
        print("  python capture_error.py python -c 'import src.utils.device_manager'")
        print("  python capture_error.py python diagnostic_check.py")
        return 1
    
    command = sys.argv[1:]
    return capture_and_parse_command(command)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
