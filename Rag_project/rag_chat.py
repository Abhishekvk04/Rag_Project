"""
PageIndex Vectorless RAG Chat
Reasoning-based retrieval using document tree structure - No Vector DB needed!
Supports multiple documents.
"""

import json
import openai
import textwrap
import os
import glob
from typing import Optional, List

# ============ Configuration ============
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
DEFAULT_MODEL = os.getenv("RAG_MODEL", "gemma2:2b")

# ============ LLM Client ============
def get_llm_client():
    """Get Ollama client (OpenAI-compatible API)"""
    return openai.OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

def call_llm(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Call the LLM and return response"""
    client = get_llm_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()

# ============ Tree Utilities ============
def load_tree(json_path: str) -> dict:
    """Load the PageIndex tree structure"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_node_mapping(tree_data: dict) -> dict:
    """Create a mapping from node_id to node for quick lookup"""
    node_map = {}
    
    def traverse(nodes):
        for node in nodes:
            if 'node_id' in node:
                node_map[node['node_id']] = node
            if 'nodes' in node:
                traverse(node['nodes'])
            if 'children' in node:
                traverse(node['children'])
    
    if 'structure' in tree_data:
        traverse(tree_data['structure'])
    elif isinstance(tree_data, list):
        traverse(tree_data)
    
    return node_map

def get_tree_summary(tree_data: dict) -> str:
    """Get a simplified tree view for the LLM to reason over"""
    lines = []
    doc_name = tree_data.get('doc_name', 'Document')
    lines.append(f"Document: {doc_name}\n")
    lines.append("Sections:")
    
    structure = tree_data.get('structure', [])
    for node in structure:
        node_id = node.get('node_id', 'N/A')
        title = node.get('title', 'Untitled')
        pages = f"Page {node.get('start_index', '?')}"
        if node.get('end_index') and node.get('end_index') != node.get('start_index'):
            pages = f"Pages {node.get('start_index')}-{node.get('end_index')}"
        summary = node.get('summary', '')[:150] + '...' if len(node.get('summary', '')) > 150 else node.get('summary', '')
        
        lines.append(f"\n[{node_id}] {title} ({pages})")
        lines.append(f"    Summary: {summary}")
    
    return '\n'.join(lines)

def print_wrapped(text: str, width: int = 80):
    """Print text wrapped to specified width"""
    print(textwrap.fill(text, width=width))

# ============ Multi-Hop Reasoning ============
def decompose_question(query: str, model: str) -> List[dict]:
    """Decompose a complex question into sub-questions for multi-hop reasoning"""
    prompt = f"""Analyze this question and determine if it requires multi-hop reasoning.
Multi-hop questions need information from multiple sources/topics to be combined.

Question: {query}

If this is a simple question that can be answered directly, return:
{{"is_multi_hop": false, "sub_questions": []}}

If this requires multi-hop reasoning, break it into 2-4 sub-questions:
{{"is_multi_hop": true, "sub_questions": ["sub-question 1", "sub-question 2", ...]}}

Reply with ONLY the JSON object, no other text."""

    response = call_llm(prompt, model)
    
    try:
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except:
        pass
    
    return {"is_multi_hop": False, "sub_questions": []}

# ============ RAG Pipeline (Multi-Document) ============
class PageIndexRAG:
    def __init__(self, tree_paths: List[str], model: str = DEFAULT_MODEL):
        """Initialize RAG with one or more document trees"""
        self.documents = {}  # doc_name -> tree_data
        self.node_maps = {}  # doc_name -> node_map
        self.model = model
        
        # Handle single path or list
        if isinstance(tree_paths, str):
            tree_paths = [tree_paths]
        
        for path in tree_paths:
            if os.path.exists(path):
                tree_data = load_tree(path)
                doc_name = tree_data.get('doc_name', os.path.basename(path))
                self.documents[doc_name] = tree_data
                self.node_maps[doc_name] = create_node_mapping(tree_data)
                print(f"✓ Loaded: {doc_name} ({len(self.node_maps[doc_name])} sections)")
        
        print(f"✓ Total documents: {len(self.documents)}")
        print(f"✓ Using model: {self.model}")
    
    def get_all_trees_summary(self) -> str:
        """Get combined tree summary for all documents"""
        all_summaries = []
        for doc_name, tree_data in self.documents.items():
            summary = get_tree_summary(tree_data)
            all_summaries.append(f"=== {doc_name} ===\n{summary}")
        return "\n\n".join(all_summaries)
    
    def tree_search(self, query: str) -> tuple:
        """Use LLM reasoning to find relevant nodes across all documents"""
        all_trees = self.get_all_trees_summary()
        
        search_prompt = f"""You are given a question and document structures with section summaries.
Your task is to identify which sections from which documents are most likely to contain the answer.

Question: {query}

Documents and their structures:
{all_trees}

Based on the section titles and summaries, which sections are most relevant?
Reply with ONLY a JSON object in this exact format (no other text):
{{"relevant_sections": [{{"doc": "document_name", "node_id": "node_id"}}], "reasoning": "brief explanation"}}
"""
        
        response = call_llm(search_prompt, self.model)
        
        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                sections = result.get('relevant_sections', [])
                reasoning = result.get('reasoning', '')
                return sections, reasoning
        except (json.JSONDecodeError, AttributeError):
            pass
        
        # Fallback: return all nodes from all docs
        print("⚠ Could not parse LLM response, using all sections")
        sections = []
        for doc_name, node_map in self.node_maps.items():
            for node_id in node_map:
                sections.append({"doc": doc_name, "node_id": node_id})
        return sections, "Using all sections as fallback"
    
    def get_context(self, sections: list) -> str:
        """Extract context from the specified sections across documents"""
        contexts = []
        for section in sections:
            # Handle both new format {"doc": ..., "node_id": ...} and legacy ["node_id"]
            if isinstance(section, dict):
                doc_name = section.get('doc', list(self.documents.keys())[0])
                node_id = section.get('node_id')
            else:
                doc_name = list(self.documents.keys())[0]
                node_id = section
            
            if doc_name in self.node_maps and node_id in self.node_maps[doc_name]:
                node = self.node_maps[doc_name][node_id]
                title = node.get('title', 'Untitled')
                summary = node.get('summary', node.get('text', ''))
                page = node.get('start_index', '?')
                contexts.append(f"[{doc_name} > {title} - Page {page}]\n{summary}")
        return "\n\n".join(contexts)
    
    def generate_answer(self, query: str, context: str) -> str:
        """Generate answer based on retrieved context"""
        doc_names = ", ".join(self.documents.keys())
        answer_prompt = f"""Answer the question based ONLY on the provided context.
If the context doesn't contain enough information, say so.
Include document name and page references when possible.

Question: {query}

Context from documents ({doc_names}):
{context}

Answer:"""
        
        return call_llm(answer_prompt, self.model)
    
    def ask(self, query: str, verbose: bool = True) -> str:
        """Full RAG pipeline: search -> retrieve -> answer"""
        if verbose:
            print(f"\n{'='*60}")
            print(f"Question: {query}")
            print('='*60)
        
        if verbose:
            print("\n🔍 Step 1: Reasoning-based tree search...")
        relevant_sections, reasoning = self.tree_search(query)
        
        if verbose:
            print(f"   Reasoning: {reasoning}")
            print(f"   Found {len(relevant_sections)} relevant section(s)")
        
        if verbose:
            print("\n📄 Step 2: Extracting context from relevant sections...")
        context = self.get_context(relevant_sections)
        
        if verbose and len(context) > 0:
            preview = context[:300] + "..." if len(context) > 300 else context
            print(f"   Context preview: {preview}")
        
        if verbose:
            print("\n💡 Step 3: Generating answer...")
        answer = self.generate_answer(query, context)
        
        if verbose:
            print(f"\n{'='*60}")
            print("ANSWER:")
            print('='*60)
            print_wrapped(answer)
            print()
        
        return answer
    
    def ask_multihop(self, query: str, verbose: bool = True) -> str:
        """Multi-hop RAG: decompose -> search each -> synthesize"""
        if verbose:
            print(f"\n{'='*60}")
            print(f"Question (Multi-Hop): {query}")
            print('='*60)
        
        # Step 1: Decompose the question
        if verbose:
            print("\n🧩 Step 1: Decomposing question...")
        decomposition = decompose_question(query, self.model)
        
        if not decomposition.get('is_multi_hop', False):
            if verbose:
                print("   → Simple question detected, using single-hop...")
            return self.ask(query, verbose=verbose)
        
        sub_questions = decomposition.get('sub_questions', [])
        if verbose:
            print(f"   Found {len(sub_questions)} sub-questions:")
            for i, sq in enumerate(sub_questions, 1):
                print(f"      {i}. {sq}")
        
        # Step 2: Answer each sub-question
        if verbose:
            print("\n🔄 Step 2: Answering sub-questions...")
        
        sub_answers = []
        for i, sub_q in enumerate(sub_questions, 1):
            if verbose:
                print(f"\n   --- Sub-question {i}: {sub_q[:50]}...")
            
            # Search for this sub-question
            relevant_sections, reasoning = self.tree_search(sub_q)
            if verbose:
                print(f"       Found {len(relevant_sections)} relevant section(s)")
            
            # Get context and generate sub-answer
            context = self.get_context(relevant_sections)
            sub_answer = self.generate_answer(sub_q, context)
            sub_answers.append({
                "question": sub_q,
                "answer": sub_answer,
                "sections": relevant_sections
            })
            
            if verbose:
                print(f"       Answer: {sub_answer[:100]}...")
        
        # Step 3: Synthesize final answer
        if verbose:
            print("\n🎯 Step 3: Synthesizing final answer...")
        
        synthesis_context = "\n\n".join([
            f"Sub-question: {sa['question']}\nAnswer: {sa['answer']}"
            for sa in sub_answers
        ])
        
        synthesis_prompt = f"""Based on the following sub-questions and their answers, provide a comprehensive answer to the original question.
Synthesize the information coherently, noting any connections or contradictions.

Original Question: {query}

Research Results:
{synthesis_context}

Synthesized Answer:"""
        
        final_answer = call_llm(synthesis_prompt, self.model)
        
        if verbose:
            print(f"\n{'='*60}")
            print("FINAL ANSWER (Multi-Hop):")
            print('='*60)
            print_wrapped(final_answer)
            print()
        
        return final_answer
    
    def ask_auto(self, query: str, verbose: bool = True) -> str:
        """Automatically detect if multi-hop is needed and use appropriate method"""
        if verbose:
            print(f"\n🤔 Analyzing question complexity...")
        
        decomposition = decompose_question(query, self.model)
        
        if decomposition.get('is_multi_hop', False):
            if verbose:
                print("   → Multi-hop reasoning required")
            return self.ask_multihop(query, verbose=verbose)
        else:
            if verbose:
                print("   → Single-hop sufficient")
            return self.ask(query, verbose=verbose)


# ============ Interactive Chat ============
def main():
    import argparse
    parser = argparse.ArgumentParser(description='PageIndex Vectorless RAG Chat (Multi-Document, Multi-Hop)')
    parser.add_argument('--tree', type=str, nargs='+', default=None,
                        help='Path(s) to PageIndex tree structure JSON file(s)')
    parser.add_argument('--tree-dir', type=str, default='./results',
                        help='Directory containing tree structure JSONs (default: ./results)')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                        help='LLM model to use')
    parser.add_argument('--query', type=str, default=None,
                        help='Single query mode')
    parser.add_argument('--multihop', action='store_true',
                        help='Force multi-hop reasoning')
    parser.add_argument('--auto', action='store_true', default=True,
                        help='Auto-detect if multi-hop is needed (default)')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  PageIndex: Vectorless, Reasoning-based RAG")
    print("  Multi-Document • Multi-Hop Reasoning")
    print("="*60 + "\n")
    
    # Collect tree paths
    tree_paths = []
    
    if args.tree:
        # Use explicitly provided tree files
        tree_paths = args.tree
    else:
        # Auto-discover from results directory
        pattern = os.path.join(args.tree_dir, '*_structure.json')
        tree_paths = glob.glob(pattern)
        if not tree_paths:
            print(f"❌ No *_structure.json files found in '{args.tree_dir}'")
            print("   Run 'python run_pageindex.py --pdf_path pdfs/<your.pdf>' first")
            return
    
    # Initialize RAG
    try:
        rag = PageIndexRAG(tree_paths, args.model)
    except Exception as e:
        print(f"❌ Error loading documents: {e}")
        return
    
    if not rag.documents:
        print("❌ No documents loaded")
        return
    
    # Determine which method to use
    def do_ask(query):
        if args.multihop:
            return rag.ask_multihop(query)
        else:
            return rag.ask_auto(query)  # Auto-detect
    
    if args.query:
        do_ask(args.query)
        return
    
    print("\n💬 Interactive mode - Commands:")
    print("   'quit' - Exit")
    print("   'docs' - List loaded documents")
    print("   'hop' - Toggle multi-hop mode")
    print("-" * 40)
    
    force_multihop = args.multihop
    
    while True:
        try:
            mode_indicator = "[multi-hop] " if force_multihop else ""
            query = input(f"\n📝 {mode_indicator}Your question: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            if query.lower() == 'docs':
                print("\n📚 Loaded documents:")
                for doc_name, tree in rag.documents.items():
                    sections = len(rag.node_maps[doc_name])
                    print(f"   • {doc_name} ({sections} sections)")
                continue
            if query.lower() == 'hop':
                force_multihop = not force_multihop
                mode = "Multi-hop (always decompose)" if force_multihop else "Auto-detect"
                print(f"   🔄 Mode: {mode}")
                continue
            if not query:
                continue
            
            if force_multihop:
                rag.ask_multihop(query)
            else:
                rag.ask_auto(query)
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break


if __name__ == "__main__":
    main()
