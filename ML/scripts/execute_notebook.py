"""Execute a Jupyter notebook using nbclient and save with stored outputs."""

import sys
import time
import asyncio
from pathlib import Path

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import nbformat
from nbclient import NotebookClient

def execute_notebook(nb_path: Path):
    print(f"Reading notebook: {nb_path}")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    client = NotebookClient(
        nb,
        timeout=1200,  # 20 minutes max
        kernel_name="python3",
        resources={"metadata": {"path": str(nb_path.parent.parent)}},
    )

    t0 = time.time()
    print("Executing notebook top to bottom...")
    client.execute()
    elapsed = time.time() - t0
    print(f"Notebook executed successfully in {elapsed:.1f}s!")

    print(f"Saving executed notebook to {nb_path}...")
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    # Calculate statistics
    total_cells = len(nb.cells)
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    cells_with_output = [c for c in code_cells if len(c.get("outputs", [])) > 0]
    file_size_bytes = nb_path.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)

    print("\n=== EXECUTION SUMMARY ===")
    print(f"Total cells: {total_cells}")
    print(f"Code cells: {len(code_cells)}")
    print(f"Code cells with stored outputs: {len(cells_with_output)} / {len(code_cells)}")
    print(f"File size: {file_size_mb:.2f} MB ({file_size_bytes:,} bytes)")

    return {
        "total_cells": total_cells,
        "code_cells": len(code_cells),
        "cells_with_output": len(cells_with_output),
        "file_size_bytes": file_size_bytes,
        "file_size_mb": round(file_size_mb, 2),
    }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = Path(sys.argv[1]).resolve()
    else:
        target = Path(__file__).resolve().parent.parent / "notebooks" / "03_EDA.ipynb"
    execute_notebook(target)

