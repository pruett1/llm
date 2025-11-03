import json

def jsonl_to_texts(filepath: str) -> list[str]:
    result = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            for key, val in obj.items():
                match key:
                    case 'desc':
                        result.append(f"<|DESC|>{val}")
                    case 'examples':
                        result.append(f"<|EXAMPLES|>{val}")
                    case 'constraints':
                        result.append(f"<|CONSTRAINTS|>{val}")
                    case 'solution':
                        result.append(f"<|OUTPUT|>{val}")
    return result

# [print(text) for text in jsonl_to_texts('./corpuses/easy_array_python_0.6_8_1755876977.jsonl')]