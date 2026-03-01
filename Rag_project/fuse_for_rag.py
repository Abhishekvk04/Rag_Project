"""
Fuse PageIndex tree structure for Vector DB RAG (optional)
Prepares hierarchical chunks with metadata fusion.
"""

import json
import os

def flatten_pageindex_tree(node, current_path=None, flattened_chunks=None, doc_name=None):
    """Recursively walks the PageIndex JSON tree to capture ancestral breadcrumbs and text content."""
    if current_path is None:
        current_path = []
    if flattened_chunks is None:
        flattened_chunks = []

    heading = node.get("title", "Document Root")
    new_path = current_path + [heading]

    content = node.get("content") or node.get("summary")
    if content:
        pages = node.get("pages", [])
        if not pages and "start_index" in node:
            start = node.get("start_index", 1)
            end = node.get("end_index", start)
            pages = list(range(start, end + 1))
        
        flattened_chunks.append({
            "hierarchy_path": new_path,
            "text": content,
            "pages": pages
        })

    children = node.get("children", node.get("nodes", []))
    for child in children:
        flatten_pageindex_tree(child, new_path, flattened_chunks, doc_name)

    return flattened_chunks

def prepare_for_vector_db(json_path):
    """Reads the JSON and fuses the metadata with the text."""
    print(f"Reading PageIndex Tree: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        tree_data = json.load(f)
    
    if isinstance(tree_data, list):
        raise ValueError(
            f"The file '{json_path}' is a log file, not a PageIndex tree structure.\n"
            "Please check the run_pageindex.py output for errors."
        )
    
    doc_name = tree_data.get("doc_name", "Document")
    if "structure" in tree_data:
        raw_chunks = []
        for item in tree_data["structure"]:
            flatten_pageindex_tree(item, [doc_name], raw_chunks, doc_name)
    else:
        raw_chunks = flatten_pageindex_tree(tree_data)

    ready_for_vector_db = []
    
    for i, chunk in enumerate(raw_chunks):
        path_string = " > ".join(chunk["hierarchy_path"])
        pages_string = f"(Pages: {', '.join(map(str, chunk['pages']))})" if chunk['pages'] else ""
        
        fused_text = f"Document Section: {path_string} {pages_string}\nContent: {chunk['text']}"
        
        ready_for_vector_db.append({
            "chunk_id": f"chunk_{i:03d}",
            "fused_text": fused_text,
            "metadata": {
                "hierarchy": chunk["hierarchy_path"],
                "pages": chunk["pages"]
            }
        })

    return ready_for_vector_db

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Fuse PageIndex tree for Vector DB')
    parser.add_argument('--input', type=str, default='./results/report_structure.json',
                        help='Input PageIndex tree JSON')
    parser.add_argument('--output', type=str, default='./results/fused_for_rag.json',
                        help='Output fused JSON for vector DB')
    args = parser.parse_args()
    
    if os.path.exists(args.input):
        fused_data = prepare_for_vector_db(args.input)
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(fused_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Successfully fused {len(fused_data)} hierarchical chunks!")
        print(f"✓ Saved to: {args.output}\n")
        
        for chunk in fused_data[:2]:
            print(f"[{chunk['chunk_id']}]")
            print(f"{chunk['fused_text'][:200]}...\n")
    else:
        print(f"❌ Could not find {args.input}")
