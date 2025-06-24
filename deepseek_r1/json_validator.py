import json

with open("dataset.jsonl", "r") as f:
    for i, line in enumerate(f, 1):
        try:
            json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON on line {i}: {e}")
