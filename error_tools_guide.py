# File: error_tools_guide.py
# Directory: ./

"""
Usage guide and examples for the error parsing tools.
"""

def show_usage_guide():
    """Display comprehensive usage guide for error parsing tools."""
    
    guide = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                           ERROR PARSING TOOLS GUIDE                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

These tools help extract relevant information from Python errors for debugging.

█ TOOL 1: parse_errors.py
  Parse error messages and get structured summaries with suggested fixes.

  Usage Options:
  
  1️⃣  Parse from stdin (interactive):
      python parse_errors.py
      # Then paste your error message and press Ctrl+D
  
  2️⃣  Parse from file:
      python parse_errors.py error_log.txt
  
  3️⃣  Parse from command output:
      python -m pytest test.py -v 2>&1 | python parse_errors.py

█ TOOL 2: capture_error.py
  Run commands and automatically parse any errors that occur.

  Usage:
      python capture_error.py <command> [args...]
  
  Examples:
      python capture_error.py python -m pytest tests/test_utils/test_device_manager.py -v
      python capture_error.py python -c "import src.utils.device_manager"
      python capture_error.py python diagnostic_check.py

█ COMMON WORKFLOWS:

  🔧 Testing Module 1:
      python capture_error.py python -m pytest tests/test_utils/test_device_manager.py -v
      
  🔧 Quick Import Check:
      python capture_error.py python -c "from src.utils.device_manager import detect_device"
      
  🔧 Running Diagnostics:
      python capture_error.py python diagnostic_check.py
      
  🔧 Parse Last Error Again:
      python parse_errors.py last_error.txt

█ ERROR TYPES SUPPORTED:
  
  • ImportError / ModuleNotFoundError
  • SyntaxError  
  • TypeError / ValueError / KeyError / AttributeError
  • RuntimeError
  • pytest test failures
  • Generic Python errors

█ OUTPUT FEATURES:

  ✓ Structured error summaries
  ✓ File locations with line numbers  
  ✓ Specific suggested fixes
  ✓ Context information
  ✓ Error logs saved to last_error.txt

█ EXAMPLE OUTPUT:

============================================================
ERROR TYPE: ImportError
MESSAGE: Cannot import 'detect_device' from 'utils.device_manager'
LOCATION: /path/to/test_file.py:19
    
SUGGESTED FIXES:
  1. Check if 'detect_device' is defined in utils.device_manager
  2. Verify the spelling of 'detect_device'
  3. Check if utils.device_manager module exists and is in PYTHONPATH
  4. Run diagnostic_check.py to verify file structure
    
CONTEXT:
  module_lookup_path: src/utils/device_manager.py, tests/test_utils/test_device_manager.py
============================================================

█ INTEGRATION WITH PROJECT:

  These tools integrate with our Phase 1 development:
  
  • Error boundaries at function entry points  
  • Comprehensive test validation before advancing
  • CPU debugging before GPU deployment
  • Structured logging with error context

█ NEXT STEPS FOR MODULE 1:

  1. Run: python capture_error.py python -m pytest tests/test_utils/test_device_manager.py -v
  2. If tests pass: ✓ Ready for Module 2 (Logging Configuration)
  3. If tests fail: Fix issues using suggested fixes from error parser
  4. Verify: python capture_error.py python diagnostic_check.py

"""
    
    print(guide)


def run_quick_demo():
    """Run a quick demonstration of the error parsing tools."""
    print("🎯 QUICK DEMO: Error Parsing Tools\n")
    
    # Show a sample error message
    sample_error = '''
Traceback (most recent call last):
  File "tests/test_utils/test_device_manager.py", line 19, in <module>
    from utils.device_manager import detect_device
  File "src/utils/__init__.py", line 8, in <module>
    from .device_manager import detect_device
ImportError: cannot import name 'detect_device' from 'utils.device_manager'
'''
    
    print("Sample error message:")
    print("-" * 40)
    print(sample_error)
    print("-" * 40)
    
    # Parse it
    from parse_errors import ErrorParser, format_error_summary
    
    parser = ErrorParser()
    summary = parser.parse_error(sample_error)
    
    print("\nParsed summary:")
    print(format_error_summary(summary))


def main():
    """Main function for the guide."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        run_quick_demo()
    else:
        show_usage_guide()


if __name__ == "__main__":
    main()
