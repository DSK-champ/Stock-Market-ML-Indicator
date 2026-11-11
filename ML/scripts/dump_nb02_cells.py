import json
from pathlib import Path

nb_path = Path("ML/notebooks/02_Define_Targets.ipynb")
with open(nb_path, encoding="utf-8") as f:
    nb = json.load(f)

for idx in [4, 5, 6, 11, 13]:
    print(f"=== CELL {idx} ===")
    print("".join(nb['cells'][idx]['source']))

