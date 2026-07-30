from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mitsui.data import list_data_files, read_table, target_columns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--preview-rows", type=int, default=3)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    inventory = list_data_files(data_dir)
    if inventory.empty:
        print(f"No data files found under {data_dir.resolve()}")
        return

    print("Data inventory:")
    print(inventory.to_string(index=False))

    for rel_path in inventory["path"]:
        path = data_dir / rel_path
        if path.suffix.lower() not in {".csv", ".parquet", ".pq"}:
            continue
        print(f"\nPreview: {rel_path}")
        try:
            df = read_table(path, nrows=args.preview_rows) if path.suffix.lower() == ".csv" else read_table(path)
        except Exception as exc:
            print(f"  Could not read file: {exc}")
            continue
        print(f"  shape preview: {df.shape}")
        targets = target_columns(df.columns)
        if targets:
            print(f"  target columns: {len(targets)} ({targets[:3]} ... {targets[-3:]})")
        print(df.head(args.preview_rows).to_string(index=False))


if __name__ == "__main__":
    main()
