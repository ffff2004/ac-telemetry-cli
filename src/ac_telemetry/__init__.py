"""Assetto Corsa replay telemetry preprocessing library."""

__version__ = "0.1.0"

from .config import ProcessingConfig
from .pipeline import preprocess_dataset

__all__ = ["ProcessingConfig", "preprocess_dataset"]
