from __future__ import annotations

from pathlib import Path

import pandas as pd


def list_data_files(data_dir: str | Path) -> pd.DataFrame:
    """Return a compact inventory of files in the official data directory.

    The inventory is intentionally metadata-only: loading every competition
    file just to inspect a data directory would waste memory in Colab.
    """
    root = Path(data_dir)
    rows = []
    for path in sorted(root.rglob("*")):
        if path.name == ".gitkeep":
            continue
        if path.is_file():
            rows.append(
                {
                    "path": str(path.relative_to(root)),
                    "size_mb": round(path.stat().st_size / 1024 / 1024, 3),
                    "suffix": path.suffix.lower(),
                }
            )
    return pd.DataFrame(rows, columns=["path", "size_mb", "suffix"])


def read_table(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read a CSV or parquet table by suffix.

    Keeping format dispatch in one place lets data-inspection scripts accept
    either format without duplicating loader selection logic.
    """
    table_path = Path(path)
    suffix = table_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(table_path, **kwargs)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(table_path, **kwargs)
    raise ValueError(f"Unsupported table format: {table_path}")


def target_columns(columns: list[str] | pd.Index) -> list[str]:
    """Return competition target columns in numeric target order.

    Lexicographic sorting would place ``target_10`` before ``target_2``.
    Numeric ordering is required by the competition inference interface.
    """
    names = [str(column) for column in columns if str(column).startswith("target_")]
    return sorted(names, key=lambda name: int(name.split("_", maxsplit=1)[1]))
