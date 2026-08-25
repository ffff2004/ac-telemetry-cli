import importlib.util
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
    """Read and write dataset tables with a Parquet-first, CSV-fallback policy."""

    def __init__(self, root: Path, requested_format: str = "auto") -> None:
        self.root = root
        if requested_format not in {"auto", "parquet", "csv"}:
            raise ValueError("storage format must be auto, parquet, or csv")
        parquet_available = importlib.util.find_spec("pyarrow") is not None
        if requested_format == "parquet" and not parquet_available:
            raise RuntimeError(
                "Parquet output requested but pyarrow is not installed. "
                "Install with: pip install 'ac-telemetry[parquet]'"
            )
        self.format = (
            "parquet"
            if requested_format == "parquet"
            or (requested_format == "auto" and parquet_available)
            else "csv"
        )

    @property
    def suffix(self) -> str:
        return ".parquet" if self.format == "parquet" else ".csv"

    def write(self, logical_name: str, frame: pd.DataFrame) -> TableRef:
        relative = Path(logical_name).with_suffix(self.suffix)
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.format == "parquet":
            frame.to_parquet(path, index=False)
        else:
            frame.to_csv(path, index=False)
        return TableRef(
            logical_name, relative.as_posix(), len(frame), len(frame.columns)
        )

    def read(self, relative_path: str) -> pd.DataFrame:
        path = self.root / relative_path
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix == ".csv":
            return pd.read_csv(path, low_memory=False)
        raise ValueError(f"Unsupported table format: {path}")
