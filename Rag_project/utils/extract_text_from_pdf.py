"""
Extract text from PDF and update structure with actual extracted text instead of summaries
"""
import json
import sys
from pathlib import Path

try:
    import pdfplumber
except ImportError:
    print("Installing pdfplumber...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pdfplumber", "-q"])
    import pdfplumber

def extract_text_from_pdf_section(pdf_path, start_page, end_page):
    """Extract text from a specific page range in the PDF"""
    extracted_text = []
    
    with pdfplumber.open(pdf_path) as pdf:
        # Convert to 0-based indexing
        for page_num in range(start_page - 1, end_page):
            if page_num < len(pdf.pages):
                page = pdf.pages[page_num]
                text = page.extract_text()
                if text:
                    extracted_text.append(text.strip())
    
    return "\n".join(extracted_text)

def main():
    # Paths
    pdf_path = Path(__file__).parent / "pdfs" / "report.pdf"
    structure_path = Path(__file__).parent / "results" / "report_structure1.json"
    output_path = Path(__file__).parent / "results" / "report_structure_with_extracted_text1.json"
    
    if not pdf_path.exists():
        print(f"Error: PDF not found at {pdf_path}")
        return
    
    if not structure_path.exists():
        print(f"Error: Structure file not found at {structure_path}")
        return
    
    # Load the structure
    with open(structure_path, 'r') as f:
        data = json.load(f)
    
    # Extract text for each section
    print("Extracting text from PDF...")
    for i, section in enumerate(data["structure"]):
        start_idx = section.get("start_index", 1)
        end_idx = section.get("end_index", 1)
        title = section.get("title", "")
        
        print(f"  [{i+1}] Extracting {title} (pages {start_idx}-{end_idx})...")
        
        extracted_text = extract_text_from_pdf_section(str(pdf_path), start_idx, end_idx)
        
        # Replace summary with extracted text, include title
        section["text"] = f"# {title}\n\n{extracted_text}" if extracted_text else f"# {title}"
        # Keep summary for reference but mark it as generated
        section["summary_type"] = "extracted_from_pdf"
    
    # Save the updated structure
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✓ Done! Extracted text saved to {output_path}")

if __name__ == "__main__":
    main()
