"""Assetto Corsa replay telemetry preprocessing library."""

__version__ = "0.1.0"

from .config import ProcessingConfig
from .events import (
    EventConfigError,
    EventDataset,
    EventInputError,
    VehicleProfile,
    detect_events,
)
from .pipeline import preprocess_dataset
from .track import TrackModel

__all__ = [
    "EventConfigError",
    "EventDataset",
    "EventInputError",
    "ProcessingConfig",
    "VehicleProfile",
    "detect_events",
    "TrackModel",
    "preprocess_dataset",
]
