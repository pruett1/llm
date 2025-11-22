import json
import time
import os

def json_to_s_exp(json_file_path: str, output_file_path: str, limit: int = -1, skip_if_exists: bool = True):
    if os.path.exists(output_file_path) and skip_if_exists:
        print(f"File {output_file_path} already exists, passing...")
        return
    start = time.time()
    def ast_to_s_exp(ast_list):
        def traverse(node):
            node_type = node['type']
            if 'value' in node:
                val = node['value'].replace('\\', '\\\\').replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t').replace('"', '\\"')
                s = f'({node_type} "{val}"'
            else:
                s = f'({node_type}'

            if 'children' in node and node['children']:
                for child_idx in node['children']:
                    if isinstance(child_idx, list):
                        for idx in child_idx:
                            s += ' ' + traverse(ast_list[idx])
                    elif isinstance(child_idx, int):
                        s += ' ' + traverse(ast_list[child_idx])
            s += ')'

            return s
        
        # stack = []
        # s = ast_list[0]
        # for char in s:
        #     if char == '(':
        #         stack.append(char)
        #     elif char == ')':
        #         if stack:
        #             stack.pop()
        #         else:
        #             raise ValueError("Unbalanced parentheses in AST")
        
        # if stack:
        #     raise ValueError("Unbalanced parentheses in AST")
        return traverse(ast_list[0])
    
    with open(json_file_path, 'r', encoding='utf-8') as json_file, open(output_file_path, 'w', encoding='utf-8') as out:
        counter = 1
        for line in json_file:
            if counter > limit and limit != -1:
                break
            line = line.strip()
            if not line:
                continue
            ast_list = json.loads(line)
            s_exp = ast_to_s_exp(ast_list)
            out.write(s_exp + '\n')
            counter += 1
            
    end = time.time()
    print(f"Converted JSON ASTs to s-expressions in {end - start:.2f} sec")

# json_to_s_exp('corpuses/python50k_eval.json', 'corpuses/test.txt', limit=100)