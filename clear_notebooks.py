#!/usr/bin/env python3
"""Clear outputs and execution counts from all notebooks in the repository."""

from __future__ import annotations

import json
from pathlib import Path


def iter_notebooks(root: Path) -> list[Path]:
    return [
        path
        for path in root.rglob("*.ipynb")
        if ".ipynb_checkpoints" not in path.parts and path.is_file()
    ]


def clear_notebook(path: Path) -> bool:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    changed = False

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        if cell.get("execution_count") is not None:
            cell["execution_count"] = None
            changed = True

        if cell.get("outputs"):
            cell["outputs"] = []
            changed = True

    if changed:
        path.write_text(
            json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )

    return changed


def main() -> int:
    root = Path(__file__).resolve().parent
    notebooks = iter_notebooks(root)

    cleared = 0
    for path in notebooks:
        if clear_notebook(path):
            cleared += 1
            print(f"Cleared {path.relative_to(root)}")

    print(f"Processed {len(notebooks)} notebook(s); cleared {cleared}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
