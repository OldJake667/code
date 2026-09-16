"""Execute the ViT-Tiny notebook cells in order without a Jupyter kernel."""

import json
import os
from pathlib import Path


os.environ.setdefault("MPLBACKEND", "Agg")

NOTEBOOK = Path(__file__).with_name("vit_tiny_from_scratch.ipynb")


def main():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}

    for index, cell in enumerate(notebook["cells"], start=1):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if source.lstrip().startswith("%pip"):
            print(f"Skipping dependency cell {index}; the project environment is ready.", flush=True)
            continue

        print(f"\n=== Running notebook cell {index} ===", flush=True)
        exec(compile(source, f"{NOTEBOOK.name}:cell-{index}", "exec"), namespace)

    print("\nNotebook execution completed successfully.", flush=True)


if __name__ == "__main__":
    main()
