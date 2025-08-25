import hou
import requests
import json
import re
import code_format_utils

ex_output = "cube_node = hou.node('/obj').createNode('geo').createNode('box', node_name='box1')"

def query_ollama(prompt):
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "applicaiton/json"}

    payload = {
        "model": "gemma3:12b",
        "prompt": f"You are a Houdini Python assistant. "
                f"User: {prompt}\n"
                f"Return only Python code using the hou API."
                f"Do not include functions or return statements."
                f"Do not include explanations, only raw hou code."
                f"Do not add any comments."
                f"Do not explain, only code."
                f"include all the variables being called."
                f"Output: {ex_output}"   
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload), stream=True)
    if response:
        print(response)
    
    code = ""

    for line in response.iter_lines():
        if line:
            data = json.loads(line.decode("utf-8"))
            if "response" in data:
                code += data["response"]
    return code.strip()

def run_prompt():
    prompt = hou.ui.readInput("Enter command: ")[1]
    if not prompt:
        return

    code = query_ollama(prompt)
    print("LLM Generated Code:\n", code)
    safe_code = code_format_utils.clean_code(code)
    print("LLM clean code:= \n", safe_code)

    try:
        exec(safe_code, {"hou": hou})
    except Exception as e:
        print(e)
    

