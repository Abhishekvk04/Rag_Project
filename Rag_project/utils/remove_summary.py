"""
Remove summary field from the enriched structure JSON
"""
import json
from pathlib import Path

def remove_summary(json_file):
    """Remove summary field from the structure"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    for section in data.get("structure", []):
        section.pop("summary", None)
    
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Removed 'summary' field from {json_file}")

if __name__ == "__main__":
    json_file = Path(__file__).parent / "results" / "report_structure_enriched.json"
    if json_file.exists():
        remove_summary(json_file)
    else:
        print(f"File not found: {json_file}")
