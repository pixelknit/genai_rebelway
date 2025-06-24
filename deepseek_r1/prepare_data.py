import pandas as pd
import json

df = pd.read_csv('data.csv')

df = df.dropna(subset=['amazon', 'twitter'])

formatted_data = []

for review in df['amazon']:
    formatted_data.append({
        "prompt": "Write an Amazon review:",
        "completion": review.strip()
        })

for tweet in df['twitter']:
    formatted_data.append({
        "prompt": "Write a tweet:",
        "completion": tweet.strip()
        })

"""
with open("dataset.jsonl", "w") as f:
    for item in formatted_data:
        json.dump(item, f)
        f.write("/n")
"""

with open("dataset.jsonl", "w") as f:
    for item in formatted_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

