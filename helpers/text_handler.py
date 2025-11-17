import json

def jsonl_to_texts(filepath: str) -> list[str]:
    result = []
    if filepath.endswith('.jsonl') is False:
        raise ValueError("Filepath must point to a .jsonl file")
    
    if 'mbpp' in filepath.lower():
        print("Detected MBPP dataset")
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                text = f"<|DESC|>{obj['text']}\n<|EXAMPLES|>{"\n".join(obj['test_list'])}\n<|OUTPUT|>{obj['code']}"
                result.append(text)
    else:
        print("Processing custom JSONL format")
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                line = ""
                for key, val in obj.items():
                    match key:
                        case 'desc':
                            line += f"<|DESC|>{val}"
                        case 'examples':
                            line += f"<|EXAMPLES|>{val}"
                        case 'constraints':
                            line += f"<|CONSTRAINTS|>{val}"
                        case 'solution':
                            line += f"<|OUTPUT|>{val}"
                result.append(line)
    return result