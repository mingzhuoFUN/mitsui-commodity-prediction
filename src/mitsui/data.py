from __future__ import annotations

from pathlib import Path

import pandas as pd


def list_data_files(data_dir: str | Path) -> pd.DataFrame:
    """返回官方数据目录的精简文件清单。

    清单只读取文件元数据；如果仅为了检查目录就加载全部竞赛文件，
    会在 Colab 中浪费大量内存。
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
    """根据扩展名读取 CSV 或 Parquet 表格。

    将格式判断集中在这里，可让数据检查脚本同时支持两种格式，
    而不必重复编写加载器选择逻辑。
    """
    table_path = Path(path)
    suffix = table_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(table_path, **kwargs)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(table_path, **kwargs)
    raise ValueError(f"Unsupported table format: {table_path}")


def target_columns(columns: list[str] | pd.Index) -> list[str]:
    """按照目标编号返回竞赛目标列。

    字符串排序会把 ``target_10`` 放在 ``target_2`` 前面；
    竞赛推理接口要求按目标编号排序。
    """
    names = [str(column) for column in columns if str(column).startswith("target_")]
    return sorted(names, key=lambda name: int(name.split("_", maxsplit=1)[1]))
