"""
preprocessing.py - LaTeX, PDF, and Text Parsing Module

Converts various input formats (LaTeX, PDF, plain text) into 
a format suitable for LLM processing.
"""

import re
import os
from typing import Optional


def latex_to_text(latex_expr: str) -> str:
    """
    Convert LaTeX expressions into plain text suitable for LLM input.
    
    Args:
        latex_expr: LaTeX string to convert
        
    Returns:
        Cleaned plain text representation
    """
    # Remove LaTeX escape sequences and special characters
    text = re.sub(r"\\\\", "", latex_expr)  # Remove double backslashes
    text = re.sub(r"\$\$|\$", "", text)  # Remove dollar signs
    text = re.sub(r"\\left|\\right", "", text)  # Remove left/right delimiters
    text = re.sub(r"\\begin\{.*?\}|\\end\{.*?\}", "", text)  # Remove environments
    text = re.sub(r"\\text\{", "", text)  # Remove \text{
    text = re.sub(r"\}", "", text)  # Remove closing braces
    text = re.sub(r"\\frac", "frac", text)  # Simplify fractions
    text = re.sub(r"\\sqrt", "sqrt", text)  # Simplify square roots
    text = re.sub(r"\\[a-z]+", "", text)  # Remove remaining LaTeX commands
    text = re.sub(r"\s+", " ", text)  # Normalize whitespace
    
    return text.strip()


def pdf_to_text(pdf_path: str) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Extracted text content
        
    Raises:
        FileNotFoundError: If PDF file doesn't exist
        ImportError: If PyPDF2 is not installed
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    try:
        import PyPDF2
    except ImportError:
        raise ImportError("PyPDF2 is required for PDF parsing. Install with: pip install PyPDF2")
    
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"[Page {page_num + 1}]\n"
                    text += page_text + "\n"
    except Exception as e:
        raise RuntimeError(f"Error reading PDF: {str(e)}")
    
    return text.strip()


def prepare_problem(input_data: str, input_type: str = "text") -> str:
    """
    Convert any input type to plain text problem format.
    
    Args:
        input_data: The input data (text, LaTeX, or PDF path)
        input_type: Type of input - "text", "latex", or "pdf"
        
    Returns:
        Plain text representation of the problem
        
    Raises:
        ValueError: If input_type is not recognized
    """
    if input_type == "latex":
        return latex_to_text(input_data)
    elif input_type == "pdf":
        return pdf_to_text(input_data)
    elif input_type == "text":
        return input_data.strip()
    else:
        raise ValueError(f"Unknown input_type: {input_type}. Use 'text', 'latex', or 'pdf'.")


def batch_prepare_problems(problems: list, input_type: str = "text") -> list:
    """
    Prepare a batch of problems.
    
    Args:
        problems: List of problem strings
        input_type: Type of input for all problems
        
    Returns:
        List of prepared problem texts
    """
    return [prepare_problem(p, input_type) for p in problems]


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text (remove extra spaces, newlines).
    
    Args:
        text: Input text
        
    Returns:
        Text with normalized whitespace
    """
    return re.sub(r"\s+", " ", text).strip()


def extract_math_expression(text: str) -> Optional[str]:
    """
    Extract mathematical expressions from text.
    
    Args:
        text: Input text
        
    Returns:
        Extracted math expression or None
    """
    # Simple pattern to match common math expressions
    pattern = r"\$\$.*?\$\$|\$.*?\$"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[0] if matches else None


if __name__ == "__main__":
    # Example usage
    example_latex = r"$\text{Compute } 2 + 3 \times 5$"
    print("LaTeX Input:", example_latex)
    print("Text Output:", latex_to_text(example_latex))
    
    example_text = "Solve for x: 2x + 5 = 13"
    print("\nText Input:", example_text)
    print("Prepared:", prepare_problem(example_text, input_type="text"))
