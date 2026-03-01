# My PageIndex RAG Project

Vectorless, Reasoning-based RAG using [PageIndex](https://github.com/VectifyAI/PageIndex).
**Supports multiple documents!**

## Project Structure

```
my_rag_project/
├── pdfs/                    # Put your PDF files here
│   └── report.pdf
├── results/                 # Generated tree structures
│   └── report_structure.json
├── logs/                    # Processing logs
├── run_pageindex.py         # Generate tree structure from PDF
├── rag_chat.py              # Vectorless RAG chat (multi-doc)
├── fuse_for_rag.py          # (Optional) Prepare for Vector DB
└── README.md
```

## Quick Start

### 1. Activate virtual environment
```bash
source ../venv/bin/activate
```

### 2. Generate PageIndex tree from PDF(s)
```bash
# Process one PDF
python run_pageindex.py --pdf_path pdfs/report.pdf --model gemma2:2b

# Process multiple PDFs
python run_pageindex.py --pdf_path pdfs/doc1.pdf --model gemma2:2b
python run_pageindex.py --pdf_path pdfs/doc2.pdf --model gemma2:2b
```

### 3. Chat with your documents (Vectorless RAG)
```bash
# Auto-loads all *_structure.json from results/
python rag_chat.py

# Single question
python rag_chat.py --query "What is the revenue growth?"

# Load specific documents
python rag_chat.py --tree results/report_structure.json results/other_structure.json

# Use different model
python rag_chat.py --model llama3.2:3b
```

### Interactive Commands
- Type your question to get answers
- Type `docs` to list loaded documents
- Type `quit` to exit

### 4. (Optional) Export for Vector DB
```bash
python fuse_for_rag.py --input results/report_structure.json --output results/fused.json
```

## How It Works

1. **Tree Generation**: PageIndex analyzes your PDF and creates a hierarchical tree structure
2. **Reasoning-based Search**: LLM reasons over ALL document trees to find relevant sections
3. **Answer Generation**: Context from relevant sections (across docs) is used to generate answers

**No Vector DB • No Chunking • Multi-Document • Human-like Retrieval**
