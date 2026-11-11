import json
from pathlib import Path

nb_path = Path("ML/notebooks/02_Define_Targets.ipynb")
with open(nb_path, encoding="utf-8") as f:
    nb = json.load(f)

print("=== CELL 1 ===")
print("".join(nb['cells'][1]['source']))

