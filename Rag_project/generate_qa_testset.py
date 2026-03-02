#!/usr/bin/env python3
"""
QA Test Set Generator using PageIndex Document Structure

Generates question-answer pairs for RAG evaluation:
- Single-hop questions (from individual sections)
- Multi-hop questions (combining multiple sections)
- Exports to JSON for evaluation

Usage:
    python generate_qa_testset.py --tree results/report_structure.json --output qa_testset.json
    python generate_qa_testset.py --tree-dir ./results --num-single 10 --num-multihop 5
"""

import os
import sys
import json
import glob
import random
import argparse
from typing import List, Dict, Any, Tuple
from datetime import datetime

# Add PageIndex to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'PageIndex'))

from pageindex.utils import get_openai_client, is_ollama_model

# ============ Configuration ============
DEFAULT_MODEL = "llama3.2:3b"

# ============ LLM Helpers ============
def call_llm(prompt: str, model: str = DEFAULT_MODEL, temperature: float = 0.7) -> str:
    """Call LLM with proper client configuration"""
    client = get_openai_client(model)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=1024
    )
    return response.choices[0].message.content.strip()


def call_llm_json(prompt: str, model: str = DEFAULT_MODEL) -> Dict:
    """Call LLM and parse JSON response"""
    response = call_llm(prompt, model, temperature=0.3)
    
    # Try to extract JSON from response
    try:
        # Handle markdown code blocks
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            response = response[start:end].strip()
        
        return json.loads(response)
    except json.JSONDecodeError:
        # Try to find JSON object in response
        start = response.find('{')
        end = response.rfind('}') + 1
        if start != -1 and end > start:
            try:
                return json.loads(response[start:end])
            except:
                pass
        return {"error": "Failed to parse JSON", "raw": response}


# ============ Document Loading ============
def load_document(tree_path: str) -> Tuple[str, Dict]:
    """Load a PageIndex tree structure"""
    with open(tree_path, 'r') as f:
        data = json.load(f)
    
    doc_name = data.get('doc_name', os.path.basename(tree_path).replace('_structure.json', ''))
    tree = data.get('structure', data)
    
    return doc_name, tree


def build_node_map(tree: Any, node_map: Dict = None, parent_path: str = "") -> Dict:
    """Build flat map of all nodes in tree"""
    if node_map is None:
        node_map = {}
    
    if isinstance(tree, dict):
        title = tree.get('title', tree.get('section_title', 'Untitled'))
        path = f"{parent_path}/{title}" if parent_path else title
        
        # Store node with its content
        content = tree.get('content', tree.get('summary', ''))
        if content:
            node_map[path] = {
                'title': title,
                'content': content,
                'start_page': tree.get('start_index', tree.get('start_page', 0)),
                'end_page': tree.get('end_index', tree.get('end_page', 0)),
                'path': path
            }
        
        # Recurse into children
        children = tree.get('children', tree.get('subsections', []))
        if isinstance(children, list):
            for child in children:
                build_node_map(child, node_map, path)
    
    elif isinstance(tree, list):
        for item in tree:
            build_node_map(item, node_map, parent_path)
    
    return node_map


# ============ Question Generation ============
def generate_single_hop_question(section: Dict, model: str) -> Dict:
    """Generate a question answerable from a single section"""
    
    prompt = f"""Based on the following document section, generate ONE specific factual question that can be answered using ONLY the information in this section.

Section Title: {section['title']}
Content:
{section['content'][:2000]}

Generate a JSON response with:
- "question": A clear, specific question
- "answer": The correct answer based on the content
- "difficulty": "easy", "medium", or "hard"
- "question_type": One of ["factual", "definition", "comparison", "reasoning", "numerical"]

Important:
- The question should require understanding the content, not just keyword matching
- The answer should be directly supported by the text
- Avoid yes/no questions

JSON Response:"""

    result = call_llm_json(prompt, model)
    
    if "error" not in result:
        result['source_section'] = section['path']
        result['source_pages'] = f"{section['start_page']}-{section['end_page']}"
        result['hop_type'] = 'single'
    
    return result


def generate_multihop_question(sections: List[Dict], model: str) -> Dict:
    """Generate a question requiring information from multiple sections"""
    
    sections_text = "\n\n---\n\n".join([
        f"Section {i+1}: {s['title']}\n{s['content'][:1500]}"
        for i, s in enumerate(sections)
    ])
    
    prompt = f"""Based on the following document sections, generate ONE question that REQUIRES combining information from MULTIPLE sections to answer correctly.

{sections_text}

Generate a JSON response with:
- "question": A question requiring synthesis from multiple sections
- "answer": The comprehensive answer combining information from sections
- "reasoning": Brief explanation of how the answer combines information
- "difficulty": "medium" or "hard"
- "question_type": One of ["comparison", "synthesis", "cause-effect", "timeline", "aggregation"]

Important:
- The question MUST require information from at least 2 sections
- Single-section answers should be WRONG or incomplete
- Focus on relationships, comparisons, or combined facts

JSON Response:"""

    result = call_llm_json(prompt, model)
    
    if "error" not in result:
        result['source_sections'] = [s['path'] for s in sections]
        result['hop_type'] = 'multi'
        result['num_hops'] = len(sections)
    
    return result


def generate_cross_document_question(doc_sections: Dict[str, List[Dict]], model: str) -> Dict:
    """Generate a question requiring information from multiple documents"""
    
    sections_text = ""
    source_docs = []
    for doc_name, sections in doc_sections.items():
        source_docs.append(doc_name)
        for s in sections:
            sections_text += f"\nDocument: {doc_name}\nSection: {s['title']}\n{s['content'][:1000]}\n---\n"
    
    prompt = f"""Based on sections from DIFFERENT documents, generate ONE question requiring information from MULTIPLE documents.

{sections_text}

Generate a JSON response with:
- "question": A question requiring cross-document synthesis
- "answer": The comprehensive answer combining information
- "reasoning": How the answer uses multiple documents
- "difficulty": "hard"
- "question_type": One of ["cross-document-comparison", "cross-document-synthesis", "cross-document-timeline"]

JSON Response:"""

    result = call_llm_json(prompt, model)
    
    if "error" not in result:
        result['source_documents'] = source_docs
        result['hop_type'] = 'cross-document'
    
    return result


# ============ Test Set Generation ============
class QATestSetGenerator:
    def __init__(self, tree_paths: List[str], model: str = DEFAULT_MODEL):
        self.model = model
        self.documents: Dict[str, Any] = {}
        self.node_maps: Dict[str, Dict] = {}
        
        for path in tree_paths:
            if os.path.exists(path):
                doc_name, tree = load_document(path)
                self.documents[doc_name] = tree
                self.node_maps[doc_name] = build_node_map(tree)
                print(f"📄 Loaded: {doc_name} ({len(self.node_maps[doc_name])} sections)")
    
    def get_random_sections(self, num: int = 1, doc_name: str = None) -> List[Dict]:
        """Get random sections, optionally from a specific document"""
        all_sections = []
        
        if doc_name and doc_name in self.node_maps:
            all_sections = list(self.node_maps[doc_name].values())
        else:
            for nm in self.node_maps.values():
                all_sections.extend(nm.values())
        
        # Filter sections with meaningful content
        valid_sections = [s for s in all_sections if len(s.get('content', '')) > 200]
        
        return random.sample(valid_sections, min(num, len(valid_sections)))
    
    def generate_testset(self, 
                         num_single: int = 10,
                         num_multihop: int = 5,
                         num_crossdoc: int = 0,
                         verbose: bool = True) -> List[Dict]:
        """Generate a complete QA test set"""
        
        testset = []
        
        # Generate single-hop questions
        if verbose:
            print(f"\n📝 Generating {num_single} single-hop questions...")
        
        for i in range(num_single):
            sections = self.get_random_sections(1)
            if sections:
                if verbose:
                    print(f"   [{i+1}/{num_single}] From: {sections[0]['title'][:40]}...")
                
                qa = generate_single_hop_question(sections[0], self.model)
                if "error" not in qa:
                    qa['id'] = f"single_{i+1}"
                    testset.append(qa)
                    if verbose:
                        print(f"      ✓ Q: {qa.get('question', 'N/A')[:50]}...")
                else:
                    if verbose:
                        print(f"      ✗ Failed to generate")
        
        # Generate multi-hop questions (within same document)
        if verbose:
            print(f"\n🔗 Generating {num_multihop} multi-hop questions...")
        
        for i in range(num_multihop):
            # Pick 2-3 sections, try to get related ones
            num_sections = random.choice([2, 2, 3])  # Bias toward 2
            sections = self.get_random_sections(num_sections)
            
            if len(sections) >= 2:
                if verbose:
                    section_names = [s['title'][:30] for s in sections]
                    print(f"   [{i+1}/{num_multihop}] Combining: {', '.join(section_names)}...")
                
                qa = generate_multihop_question(sections, self.model)
                if "error" not in qa:
                    qa['id'] = f"multihop_{i+1}"
                    testset.append(qa)
                    if verbose:
                        print(f"      ✓ Q: {qa.get('question', 'N/A')[:50]}...")
                else:
                    if verbose:
                        print(f"      ✗ Failed to generate")
        
        # Generate cross-document questions (if multiple docs)
        if num_crossdoc > 0 and len(self.documents) > 1:
            if verbose:
                print(f"\n🌐 Generating {num_crossdoc} cross-document questions...")
            
            doc_names = list(self.documents.keys())
            for i in range(num_crossdoc):
                # Pick 2 random documents
                selected_docs = random.sample(doc_names, min(2, len(doc_names)))
                doc_sections = {}
                
                for doc in selected_docs:
                    sections = self.get_random_sections(1, doc)
                    if sections:
                        doc_sections[doc] = sections
                
                if len(doc_sections) >= 2:
                    if verbose:
                        print(f"   [{i+1}/{num_crossdoc}] Documents: {', '.join(doc_sections.keys())}")
                    
                    qa = generate_cross_document_question(doc_sections, self.model)
                    if "error" not in qa:
                        qa['id'] = f"crossdoc_{i+1}"
                        testset.append(qa)
                        if verbose:
                            print(f"      ✓ Q: {qa.get('question', 'N/A')[:50]}...")
        
        return testset
    
    def generate_from_section_pairs(self, verbose: bool = True) -> List[Dict]:
        """Generate multi-hop questions from semantically related sections"""
        
        testset = []
        
        # Find section pairs that might have relationships
        for doc_name, node_map in self.node_maps.items():
            sections = list(node_map.values())
            
            # Group sections by their parent path
            parent_groups = {}
            for s in sections:
                parts = s['path'].rsplit('/', 1)
                parent = parts[0] if len(parts) > 1 else "root"
                if parent not in parent_groups:
                    parent_groups[parent] = []
                parent_groups[parent].append(s)
            
            # Generate questions from sibling sections
            for parent, siblings in parent_groups.items():
                if len(siblings) >= 2:
                    pair = random.sample(siblings, 2)
                    if verbose:
                        print(f"   Sibling pair: {pair[0]['title'][:30]} + {pair[1]['title'][:30]}")
                    
                    qa = generate_multihop_question(pair, self.model)
                    if "error" not in qa:
                        qa['id'] = f"sibling_{len(testset)+1}"
                        qa['relationship'] = 'sibling_sections'
                        testset.append(qa)
        
        return testset


def save_testset(testset: List[Dict], output_path: str):
    """Save test set to JSON"""
    
    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_questions": len(testset),
            "single_hop": len([q for q in testset if q.get('hop_type') == 'single']),
            "multi_hop": len([q for q in testset if q.get('hop_type') == 'multi']),
            "cross_doc": len([q for q in testset if q.get('hop_type') == 'cross-document'])
        },
        "questions": testset
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Saved {len(testset)} questions to: {output_path}")


# ============ Main ============
def main():
    parser = argparse.ArgumentParser(description='Generate QA Test Set from PageIndex Documents')
    parser.add_argument('--tree', type=str, nargs='+', default=None,
                        help='Path(s) to PageIndex tree structure JSON file(s)')
    parser.add_argument('--tree-dir', type=str, default='./results',
                        help='Directory containing tree structure JSONs')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                        help='LLM model to use')
    parser.add_argument('--output', type=str, default='qa_testset1.json',
                        help='Output JSON file path')
    parser.add_argument('--num-single', type=int, default=10,
                        help='Number of single-hop questions')
    parser.add_argument('--num-multihop', type=int, default=5,
                        help='Number of multi-hop questions')
    parser.add_argument('--num-crossdoc', type=int, default=0,
                        help='Number of cross-document questions')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  QA Test Set Generator")
    print("  Using PageIndex Document Structure")
    print("="*60)
    
    # Collect tree paths
    tree_paths = []
    if args.tree:
        tree_paths = args.tree
    else:
        pattern = os.path.join(args.tree_dir, '*_structure1.json')
        tree_paths = glob.glob(pattern)
        if not tree_paths:
            print(f"\n❌ No *_structure1.json files found in '{args.tree_dir}'")
            return
    
    print(f"\n📚 Loading {len(tree_paths)} document(s)...")
    
    # Initialize generator
    generator = QATestSetGenerator(tree_paths, args.model)
    
    if not generator.documents:
        print("❌ No documents loaded")
        return
    
    # Generate test set
    testset = generator.generate_testset(
        num_single=args.num_single,
        num_multihop=args.num_multihop,
        num_crossdoc=args.num_crossdoc
    )
    
    # Save results
    save_testset(testset, args.output)
    
    # Print summary
    print("\n" + "="*60)
    print("📊 Summary:")
    print(f"   Single-hop questions: {len([q for q in testset if q.get('hop_type') == 'single'])}")
    print(f"   Multi-hop questions:  {len([q for q in testset if q.get('hop_type') == 'multi'])}")
    print(f"   Cross-doc questions:  {len([q for q in testset if q.get('hop_type') == 'cross-document'])}")
    print("="*60)


if __name__ == "__main__":
    main()
