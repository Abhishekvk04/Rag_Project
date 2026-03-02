#!/usr/bin/env python3
"""
Main orchestration script for RAG Project
Controls the execution flow of various scripts with user confirmation after each step
"""
import os
import sys
import subprocess
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()
VENV_PYTHON = PROJECT_ROOT.parent / "venv" / "bin" / "python"

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")

def ask_continue(script_name="Next script"):
    """Ask user if they want to continue"""
    while True:
        response = input(f"\n✓ {script_name} completed!\nDo you want to continue? (yes/no): ").strip().lower()
        if response in ['yes', 'y']:
            return True
        elif response in ['no', 'n']:
            return False
        else:
            print("Please enter 'yes' or 'no'")

def run_script(script_name, args, description):
    """Run a script and handle errors"""
    print_header(description)
    
    script_path = PROJECT_ROOT / script_name
    
    if not script_path.exists():
        print(f"❌ Error: {script_name} not found!")
        return False
    
    try:
        cmd = [str(VENV_PYTHON), str(script_path)] + args
        print(f"Running: {' '.join(cmd)}\n")
        
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {script_name}: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Main orchestration"""
    print("\n" + "🎯 RAG PROJECT ORCHESTRATOR ".center(70, "="))
    print("Control the execution flow of RAG project scripts")
    print("="*70)
    
    # Check if PDF file exists
    pdf_path = PROJECT_ROOT / "pdfs" / "report.pdf"
    if not pdf_path.exists():
        print(f"\n❌ Error: PDF file not found at {pdf_path}")
        print("Please add your PDF to the pdfs/ folder first")
        return
    
    # Get model choice from user
    print("\n📋 Available Models:")
    print("  1. mistral:7b (⭐ RECOMMENDED - best hierarchical extraction)")
    print("  2. llama3.2:3b (good, faster)")
    print("  3. gemma2:2b (fast, basic)")
    print("  4. Custom model (enter name)")
    
    model_choice = input("\nSelect model (1-4) [default: 1]: ").strip() or "1"
    
    model_map = {
        "1": "mistral:7b",
        "2": "llama3.2:3b",
        "3": "gemma2:2b",
    }
    
    if model_choice in model_map:
        model = model_map[model_choice]
    else:
        model = input("Enter model name: ").strip() or "mistral:7b"
    
    print(f"\n✅ Selected model: {model}")
    
    # Step 1: Run PageIndex
    if not run_script(
        "run_pageindex.py",
        ["--pdf_path", "pdfs/report.pdf", "--model", model],
        "STEP 1: Generate PageIndex Structure and Extract Text"
    ):
        print("\n⚠️  PageIndex generation failed. Stopping...")
        return
    
    if not ask_continue("PageIndex generation"):
        print("\n❌ Process stopped by user")
        return
    
    # Step 2: Fuse for RAG (if exists)
    if (PROJECT_ROOT / "fuse_for_rag.py").exists():
        if run_script(
            "fuse_for_rag.py",
            [],
            "STEP 2: Fuse Data for RAG"
        ):
            if not ask_continue("Fuse for RAG"):
                print("\n❌ Process stopped by user")
                return
        else:
            print("\n⚠️  Fuse for RAG failed (continuing anyway)")
    
    # Step 3: Generate QA Test Set (if exists)
    if (PROJECT_ROOT / "generate_qa_testset.py").exists():
        if run_script(
            "generate_qa_testset.py",
            [],
            "STEP 3: Generate QA Test Set"
        ):
            if not ask_continue("QA Test Set generation"):
                print("\n❌ Process stopped by user")
                return
        else:
            print("\n⚠️  QA Test Set generation failed (continuing anyway)")
    
    # Step 4: RAG Chat (if exists)
    if (PROJECT_ROOT / "rag_chat.py").exists():
        print_header("STEP 4: RAG Chat Interface")
        print("Ready to start RAG Chat?\n")
        
        if ask_continue("Launch RAG Chat"):
            try:
                run_script(
                    "rag_chat.py",
                    [],
                    "RAG Chat Interface"
                )
            except KeyboardInterrupt:
                print("\n\n❌ Chat interrupted by user")
                return
    
    # Final summary
    print_header("✅ PIPELINE COMPLETE!")
    print("All steps completed successfully!")
    print(f"\nOutputs saved in: {PROJECT_ROOT}/results/")
    print("\nKey files:")
    print("  - report_structure_enriched.json (PageIndex structure with extracted text)")
    print("  - qa_testset.json (QA test set if generated)")
    print("  - Chat logs in logs/ folder")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)
