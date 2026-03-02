"""
Remove unnecessary keys from the enriched structure JSON
"""
import json
from pathlib import Path

def clean_structure(json_file):
    """Remove unwanted keys from the structure"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    keys_to_remove = ["start_index", "end_index", "text_type", "extraction_status"]
    
    for section in data.get("structure", []):
        for key in keys_to_remove:
            section.pop(key, None)
    
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Cleaned {json_file}")
    print(f"  Removed keys: {', '.join(keys_to_remove)}")

if __name__ == "__main__":
    json_file = Path(__file__).parent / "results" / "report_structure_enriched1.json"
    if json_file.exists():
        clean_structure(json_file)
    else:
        print(f"File not found: {json_file}")
