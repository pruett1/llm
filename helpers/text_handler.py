import json

def jsonl_to_texts(filepath: str) -> list[str]:
    result = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            for key, val in obj.items():
                if key == 'solution':
                    result.append(f"<|OUTPUT|>{val}")
                else:
                    result.append(str(val))
    return result

# [print(text) for text in jsonl_to_texts('../corpuses/easy_array_python_0.6_8_1755876977.jsonl')]