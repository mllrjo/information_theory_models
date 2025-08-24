# File: parse_errors.py
# Directory: ./

"""
Error message parser to extract relevant information for debugging.
Designed to parse pytest, import, and other Python errors into structured summaries.
"""

import re
import sys
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ErrorSummary:
    """Structured representation of an error."""
    error_type: str
    primary_message: str
    file_location: Optional[str]
    line_number: Optional[int]
    suggested_fixes: List[str]
    context: Dict[str, str]


class ErrorParser:
    """Parse different types of Python error messages."""
    
    def __init__(self):
        self.parsers = {
            'ImportError': self._parse_import_error,
            'ModuleNotFoundError': self._parse_module_error,
            'SyntaxError': self._parse_syntax_error,
            'TypeError': self._parse_type_error,
            'ValueError': self._parse_value_error,
            'KeyError': self._parse_key_error,
            'AttributeError': self._parse_attribute_error,
            'pytest': self._parse_pytest_error,
            'RuntimeError': self._parse_runtime_error
        }
    
    def parse_error(self, error_text: str) -> ErrorSummary:
        """
        Parse error text and return structured summary.
        
        Args:
            error_text: Full error message text
            
        Returns:
            ErrorSummary with extracted information
        """
        # Detect error type
        error_type = self._detect_error_type(error_text)
        
        # Use appropriate parser
        if error_type in self.parsers:
            return self.parsers[error_type](error_text)
        else:
            return self._parse_generic_error(error_text)
    
    def _detect_error_type(self, text: str) -> str:
        """Detect the primary error type from text."""
        # Check for pytest first (special case)
        if 'test session starts' in text or 'FAILED' in text or 'collected' in text:
            return 'pytest'
        
        # Check for specific error types
        error_patterns = [
            (r'ImportError:', 'ImportError'),
            (r'ModuleNotFoundError:', 'ModuleNotFoundError'), 
            (r'SyntaxError:', 'SyntaxError'),
            (r'TypeError:', 'TypeError'),
            (r'ValueError:', 'ValueError'),
            (r'KeyError:', 'KeyError'),
            (r'AttributeError:', 'AttributeError'),
            (r'RuntimeError:', 'RuntimeError'),
            (r'FileNotFoundError:', 'FileNotFoundError'),
            (r'PermissionError:', 'PermissionError')
        ]
        
        for pattern, error_type in error_patterns:
            if re.search(pattern, text):
                return error_type
        
        return 'Unknown'
    
    def _extract_file_location(self, text: str) -> Tuple[Optional[str], Optional[int]]:
        """Extract file path and line number from error text."""
        # Pattern: /path/to/file.py:123: in function_name
        pattern = r'([/\w\-_.]+\.py):(\d+):'
        match = re.search(pattern, text)
        if match:
            return match.group(1), int(match.group(2))
        
        # Pattern: File "/path/to/file.py", line 123
        pattern = r'File "([^"]+)", line (\d+)'
        match = re.search(pattern, text)
        if match:
            return match.group(1), int(match.group(2))
        
        return None, None
    
    def _parse_import_error(self, text: str) -> ErrorSummary:
        """Parse ImportError messages."""
        file_path, line_num = self._extract_file_location(text)
        
        # Extract the import statement and missing module
        import_match = re.search(r"cannot import name '([^']+)' from '([^']+)'", text)
        if import_match:
            missing_name = import_match.group(1)
            module_name = import_match.group(2)
            message = f"Cannot import '{missing_name}' from '{module_name}'"
            
            fixes = [
                f"Check if '{missing_name}' is defined in {module_name}",
                f"Verify the spelling of '{missing_name}'",
                f"Check if {module_name} module exists and is in PYTHONPATH",
                "Run diagnostic_check.py to verify file structure"
            ]
        else:
            message = "Import failed"
            fixes = [
                "Check file paths and module structure",
                "Verify PYTHONPATH is set correctly",
                "Check for circular imports"
            ]
        
        return ErrorSummary(
            error_type='ImportError',
            primary_message=message,
            file_location=file_path,
            line_number=line_num,
            suggested_fixes=fixes,
            context={'module_lookup_path': self._extract_paths(text)}
        )
    
    def _parse_module_error(self, text: str) -> ErrorSummary:
        """Parse ModuleNotFoundError messages."""
        file_path, line_num = self._extract_file_location(text)
        
        module_match = re.search(r"No module named '([^']+)'", text)
        if module_match:
            missing_module = module_match.group(1)
            message = f"Module '{missing_module}' not found"
            
            fixes = [
                f"Install module: pip install {missing_module}",
                f"Check if {missing_module} is in requirements.txt",
                "Verify virtual environment is activated",
                "Check module name spelling"
            ]
        else:
            message = "Module not found"
            fixes = ["Check module installation and PYTHONPATH"]
        
        return ErrorSummary(
            error_type='ModuleNotFoundError',
            primary_message=message,
            file_location=file_path,
            line_number=line_num,
            suggested_fixes=fixes,
            context={}
        )
    
    def _parse_syntax_error(self, text: str) -> ErrorSummary:
        """Parse SyntaxError messages."""
        file_path, line_num = self._extract_file_location(text)
        
        # Extract the actual syntax error details
        syntax_match = re.search(r'SyntaxError: (.+)', text)
        message = syntax_match.group(1) if syntax_match else "Syntax error"
        
        fixes = [
            "Check for missing parentheses, brackets, or quotes",
            "Verify proper indentation",
            "Check for missing colons after if/for/def statements",
            f"Review code around line {line_num}" if line_num else "Review recent changes"
        ]
        
        return ErrorSummary(
            error_type='SyntaxError',
            primary_message=message,
            file_location=file_path,
            line_number=line_num,
            suggested_fixes=fixes,
            context={}
        )
    
    def _parse_type_error(self, text: str) -> ErrorSummary:
        """Parse TypeError messages."""
        file_path, line_num = self._extract_file_location(text)
        
        type_match = re.search(r'TypeError: (.+)', text)
        message = type_match.group(1) if type_match else "Type error"
        
        fixes = [
            "Check function argument types",
            "Verify variable types match expected types",
            "Check for None values where objects expected",
            "Review function calls and parameters"
        ]
        
        return ErrorSummary(
            error_type='TypeError',
            primary_message=message,
            file_location=file_path,
            line_number=line_num,
            suggested_fixes=fixes,
            context={}
        )
    
    def _parse_value_error(self, text: str) -> ErrorSummary:
        """Parse ValueError messages."""
        file_path, line_num = self._extract_file_location(text)
        
        value_match = re.search(r'ValueError: (.+)', text)
        message = value_match.group(1) if value_match else "Value error"
        
        fixes = [
            "Check input values are in valid range",
            "Verify data format and types",
            "Check for empty or invalid inputs",
            "Review validation logic"
        ]
        
        return ErrorSummary(
            error_type='ValueError',
            primary_message=message,
            file_location=file_path,
            line_number=line_num,
            suggested_fixes=fixes,
            context={}
        )
    
    def _parse_key_error(self, text: str) -> ErrorSummary:
        """Parse KeyError messages."""
        file_path, line_num = self._extract_file_location(text)
        
        key_match = re.search(r"KeyError: '([^']+)'", text)
        if key_match:
            missing_key = key_match.group(1)
            message = f"Key '{missing_key}' not found"
            fixes = [
                f"Check if key '{missing_key}' exists in dictionary",
                f"Use dict.get('{missing_key}', default) for safe access",
                "Verify dictionary structure matches expectations"
            ]
        else:
            message = "Key not found in dictionary"
            fixes = ["Check dictionary keys and structure"]
        
        return ErrorSummary(
            error_type='KeyError',
            primary_message=message,
            file_location=file_path,
            line_number=line_num,
            suggested_fixes=fixes,
            context={}
        )
    
    def _parse_attribute_error(self, text: str) -> ErrorSummary:
        """Parse AttributeError messages."""
        file_path, line_num = self._extract_file_location(text)
        
        attr_match = re.search(r"'([^']+)' object has no attribute '([^']+)'", text)
        if attr_match:
            obj_type = attr_match.group(1)
            attr_name = attr_match.group(2)
            message = f"'{obj_type}' has no attribute '{attr_name}'"
            fixes = [
                f"Check if '{attr_name}' is spelled correctly",
                f"Verify {obj_type} object has the expected methods/attributes",
                "Check object initialization",
                "Review object type - might be different than expected"
            ]
        else:
            message = "Attribute not found"
            fixes = ["Check attribute names and object types"]
        
        return ErrorSummary(
            error_type='AttributeError',
            primary_message=message,
            file_location=file_path,
            line_number=line_num,
            suggested_fixes=fixes,
            context={}
        )
    
    def _parse_pytest_error(self, text: str) -> ErrorSummary:
        """Parse pytest test failure messages."""
        # Count failures and errors
        failed_match = re.search(r'(\d+) failed', text)
        error_match = re.search(r'(\d+) error', text)
        passed_match = re.search(r'(\d+) passed', text)
        
        failed_count = int(failed_match.group(1)) if failed_match else 0
        error_count = int(error_match.group(1)) if error_match else 0
        passed_count = int(passed_match.group(1)) if passed_match else 0
        
        if error_count > 0:
            message = f"{error_count} test errors, {failed_count} failures, {passed_count} passed"
        elif failed_count > 0:
            message = f"{failed_count} test failures, {passed_count} passed"
        else:
            message = f"{passed_count} tests passed"
        
        # Extract specific test failure details
        file_path = None
        line_num = None
        
        # Look for FAILED lines
        failed_tests = re.findall(r'FAILED ([^:]+)::([^:]+)', text)
        
        fixes = []
        if error_count > 0:
            fixes.extend([
                "Check test imports and dependencies",
                "Verify test environment setup",
                "Run diagnostic_check.py first"
            ])
        if failed_count > 0:
            fixes.extend([
                "Review failing test assertions",
                "Check test data and expected results",
                "Verify function implementations match test expectations"
            ])
        
        context = {
            'failed_count': str(failed_count),
            'error_count': str(error_count),
            'passed_count': str(passed_count),
            'failed_tests': str(failed_tests) if failed_tests else 'none'
        }
        
        return ErrorSummary(
            error_type='pytest',
            primary_message=message,
            file_location=file_path,
            line_number=line_num,
            suggested_fixes=fixes,
            context=context
        )
    
    def _parse_runtime_error(self, text: str) -> ErrorSummary:
        """Parse RuntimeError messages."""
        file_path, line_num = self._extract_file_location(text)
        
        runtime_match = re.search(r'RuntimeError: (.+)', text)
        message = runtime_match.group(1) if runtime_match else "Runtime error"
        
        fixes = [
            "Check system resources and permissions",
            "Verify input data integrity",
            "Check for hardware/driver issues",
            "Review error context for specific guidance"
        ]
        
        return ErrorSummary(
            error_type='RuntimeError',
            primary_message=message,
            file_location=file_path,
            line_number=line_num,
            suggested_fixes=fixes,
            context={}
        )
    
    def _parse_generic_error(self, text: str) -> ErrorSummary:
        """Parse unknown error types."""
        file_path, line_num = self._extract_file_location(text)
        
        # Try to extract any error message
        lines = text.strip().split('\n')
        message = "Unknown error"
        for line in reversed(lines):
            if line.strip() and not line.startswith(' '):
                message = line.strip()
                break
        
        return ErrorSummary(
            error_type='Unknown',
            primary_message=message,
            file_location=file_path,
            line_number=line_num,
            suggested_fixes=["Review full error message for details"],
            context={}
        )
    
    def _extract_paths(self, text: str) -> str:
        """Extract file paths from error text."""
        paths = re.findall(r'[/\w\-_.]+\.py', text)
        return ', '.join(set(paths)) if paths else 'none found'


def format_error_summary(summary: ErrorSummary) -> str:
    """Format error summary for easy reading."""
    output = []
    output.append("=" * 60)
    output.append(f"ERROR TYPE: {summary.error_type}")
    output.append(f"MESSAGE: {summary.primary_message}")
    
    if summary.file_location:
        location = summary.file_location
        if summary.line_number:
            location += f":{summary.line_number}"
        output.append(f"LOCATION: {location}")
    
    if summary.suggested_fixes:
        output.append("\nSUGGESTED FIXES:")
        for i, fix in enumerate(summary.suggested_fixes, 1):
            output.append(f"  {i}. {fix}")
    
    if summary.context:
        output.append("\nCONTEXT:")
        for key, value in summary.context.items():
            output.append(f"  {key}: {value}")
    
    output.append("=" * 60)
    return '\n'.join(output)


def main():
    """Main function to parse error from stdin or file."""
    if len(sys.argv) > 1:
        # Read from file
        try:
            with open(sys.argv[1], 'r') as f:
                error_text = f.read()
        except FileNotFoundError:
            print(f"Error: File '{sys.argv[1]}' not found")
            return
    else:
        # Read from stdin
        print("Paste your error message (press Ctrl+D when done):")
        error_text = sys.stdin.read()
    
    if not error_text.strip():
        print("No error message provided")
        return
    
    parser = ErrorParser()
    summary = parser.parse_error(error_text)
    
    print(format_error_summary(summary))


if __name__ == "__main__":
    main()
