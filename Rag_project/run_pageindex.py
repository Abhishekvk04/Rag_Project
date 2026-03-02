"""
PageIndex RAG Project - Run PageIndex
Wrapper script to run PageIndex from outside the library folder.
Includes text extraction from PDF for complete RAG output.
"""
import sys
import os
import json
import argparse
from pathlib import Path

# Add PageIndex to path
PAGEINDEX_DIR = os.path.join(os.path.dirname(__file__), '..', 'PageIndex')
sys.path.insert(0, PAGEINDEX_DIR)

from pageindex import page_index_main
from pageindex.utils import ConfigLoader

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

def extract_text_from_pdf_section(pdf_path, start_page, end_page):
    """Extract text from a specific page range in the PDF"""
    if pdfplumber is None:
        print("Warning: pdfplumber not installed. Text extraction skipped.")
        return ""
    
    extracted_text = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Convert to 0-based indexing
            for page_num in range(start_page - 1, end_page):
                if page_num < len(pdf.pages):
                    page = pdf.pages[page_num]
                    text = page.extract_text()
                    if text:
                        extracted_text.append(text.strip())
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""
    
    return "\n".join(extracted_text)

def enrich_structure_with_text(structure_file, pdf_path):
    """Add extracted text to the PageIndex structure"""
    try:
        with open(structure_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading structure file: {e}")
        return
    
    if "structure" not in data:
        print("No structure found in file")
        return
    
    print("\nEnriching structure with extracted text...")
    
    for i, section in enumerate(data["structure"]):
        start_idx = section.get("start_index", 1)
        end_idx = section.get("end_index", 1)
        title = section.get("title", "")
        
        print(f"  [{i+1}] Extracting text for '{title}' (pages {start_idx}-{end_idx})...")
        
        extracted_text = extract_text_from_pdf_section(pdf_path, start_idx, end_idx)
        
        # Add extracted text to section
        if extracted_text:
            section["text"] = f"# {title}\n\n{extracted_text}"
        else:
            section["text"] = f"# {title}"
    
    # Clean up unwanted keys
    keys_to_remove = ["start_index", "end_index", "text_type", "extraction_status", "summary"]
    for section in data.get("structure", []):
        for key in keys_to_remove:
            section.pop(key, None)
    
    # Save enriched structure
    output_file = str(Path(structure_file).parent / f"{Path(structure_file).stem}_enriched1.json")
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Enriched structure saved to {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Generate PageIndex tree structure from PDF with extracted text')
    parser.add_argument('--pdf_path', type=str, required=True, help='Path to the PDF file')
    parser.add_argument('--model', type=str, default='gemma2:2b', help='LLM model to use')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory for results')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Log directory')
    parser.add_argument('--skip_text_extraction', action='store_true', help='Skip text extraction step')
    args = parser.parse_args()
    
    # Ensure output directories exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Load config using ConfigLoader
    config_path = os.path.join(PAGEINDEX_DIR, 'pageindex', 'config.yaml')
    loader = ConfigLoader(config_path)
    opt = loader.load({'model': args.model})
    
    # Run PageIndex
    print(f"Processing: {args.pdf_path}")
    print(f"Model: {args.model}")
    print(f"Output: {args.output_dir}")
    
    result = page_index_main(args.pdf_path, opt)
    print("✓ PageIndex structure generation complete!")
    
    # Extract and enrich with text
    if not args.skip_text_extraction:
        pdf_basename = Path(args.pdf_path).stem
        structure_file = os.path.join(args.output_dir, f"{pdf_basename}_structure1.json")
        
        if os.path.exists(structure_file):
            enrich_structure_with_text(structure_file, args.pdf_path)
        else:
            print(f"Warning: Structure file not found at {structure_file}")
    
    print("\n✅ Done! Check the results folder for complete output.")

if __name__ == "__main__":
    main()
