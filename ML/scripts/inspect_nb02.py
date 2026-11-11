import json
from pathlib import Path

nb_path = Path("ML/notebooks/02_Define_Targets.ipynb")
with open(nb_path, encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for i, c in enumerate(nb['cells']):
    ctype = c['cell_type']
    first_line = "".join(c.get('source', []))[:80].replace("\n", " ")
    print(f"Cell {i} [{ctype}]: {first_line}")

