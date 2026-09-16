"""Execute the pretrained DINOv2 notebook cells in order."""

import json
import os
from pathlib import Path


os.environ.setdefault("MPLBACKEND", "Agg")

NOTEBOOK = Path(__file__).with_name("DINOv2_pretrained.ipynb")


def main():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}

    for index, cell in enumerate(notebook["cells"], start=1):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        print(f"\n=== Running notebook cell {index} ===", flush=True)
        exec(compile(source, f"{NOTEBOOK.name}:cell-{index}", "exec"), namespace)

    print("\nNotebook execution completed successfully.", flush=True)


if __name__ == "__main__":
    main()
