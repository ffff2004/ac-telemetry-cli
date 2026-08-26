from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True, slots=True)
class TableRef:
    name: str
    relative_path: str
    rows: int
    columns: int


class DatasetStorage:
    """Read and write dataset tables as Parquet files."""

    def __init__(self, root: Path) -> None:
        self.root = root

    @property
    def suffix(self) -> str:
        return ".parquet"

    def write(self, logical_name: str, frame: pd.DataFrame) -> TableRef:
        relative = Path(logical_name).with_suffix(self.suffix)
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        return TableRef(
            logical_name, relative.as_posix(), len(frame), len(frame.columns)
        )

    def read(self, relative_path: str) -> pd.DataFrame:
        return pd.read_parquet(self.root / relative_path)
