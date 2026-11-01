import json
from pathlib import Path

def validate_bio_file(file_path: str):
    print(f"Validating {file_path}...")
    error_count = 0
    
    path = Path(file_path)
    if not path.exists():
        print(f"  -> File not found: {file_path}\n")
        return

    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue
            
            data = json.loads(line)
            tokens = data.get("tokens", [])
            tags = data.get("tags", [])
            
            prev_tag_type = 'O'
            prev_tag_class = None
            
            for token_idx, (token, tag) in enumerate(zip(tokens, tags)):
                if tag in ['O', '<PAD>', '<START>', '<STOP>']:
                    prev_tag_type = 'O'
                    prev_tag_class = None
                    continue
                    
                try:
                    tag_type, tag_class = tag.split('-', 1)
                except ValueError:
                    print(f"  Line {line_idx}, Token '{token}': Malformed tag '{tag}'")
                    error_count += 1
                    continue

                if tag_type == 'I':
                    # An 'I' tag MUST be preceded by a 'B' or 'I' tag of the EXACT SAME class.
                    if prev_tag_class != tag_class:
                        print(f"  Line {line_idx}, Token '{token}': '{tag}' illegally follows '{prev_tag_type}-{prev_tag_class}'")
                        error_count += 1
                
                prev_tag_type = tag_type
                prev_tag_class = tag_class

    if error_count == 0:
        print("  -> Passed! No BIO errors found.\n")
    else:
        print(f"  -> Failed. Found {error_count} BIO sequence violations.\n")

if __name__ == "__main__":
    validate_bio_file("data/processed/train.jsonl")
    validate_bio_file("data/processed/val.jsonl")
    validate_bio_file("data/processed/test.jsonl")
