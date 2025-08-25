import re

def clean_code(code: str) -> str:
    code = re.sub(f"^```[a-z]*","",code.strip(), flags=re.MULTILINE)
    code = re.sub(r"```$", "", code.strip(), flags=re.MULTILINE)
    return code.strip()