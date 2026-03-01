"""
PageIndex RAG Project - Run PageIndex
Wrapper script to run PageIndex from outside the library folder.
"""
import sys
import os
import argparse

# Add PageIndex to path
PAGEINDEX_DIR = os.path.join(os.path.dirname(__file__), '..', 'PageIndex')
sys.path.insert(0, PAGEINDEX_DIR)

from pageindex import page_index_main, load_config

def main():
    parser = argparse.ArgumentParser(description='Generate PageIndex tree structure from PDF')
    parser.add_argument('--pdf_path', type=str, required=True, help='Path to the PDF file')
    parser.add_argument('--model', type=str, default='gemma2:2b', help='LLM model to use')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory for results')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Log directory')
    args = parser.parse_args()
    
    # Ensure output directories exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Load config and override settings
    opt = load_config(os.path.join(PAGEINDEX_DIR, 'pageindex', 'config.yaml'))
    opt.model = args.model
    opt.output_dir = args.output_dir
    opt.log_dir = args.log_dir
    
    # Run PageIndex
    print(f"Processing: {args.pdf_path}")
    print(f"Model: {args.model}")
    print(f"Output: {args.output_dir}")
    
    result = page_index_main(args.pdf_path, opt)
    print("\n✓ Done! Check the results folder.")

if __name__ == "__main__":
    main()
