import json

def jsonl_to_texts(filepath: str) -> list[str]:
    result = []
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

# [print(text) for text in jsonl_to_texts('./corpuses/easy_array_python_0.6_8_1755876977.jsonl')]