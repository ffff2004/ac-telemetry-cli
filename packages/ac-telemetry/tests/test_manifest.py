import pytest
from ac_telemetry.manifest import DATASET_SCHEMA_VERSION, require_compatible_schema


def test_current_schema_is_compatible() -> None:
    assert DATASET_SCHEMA_VERSION == "5"
    require_compatible_schema([{"schema_version": DATASET_SCHEMA_VERSION}])


def test_incompatible_schema_is_rejected() -> None:
    with pytest.raises(ValueError, match="schema version"):
        require_compatible_schema([{"schema_version": "3"}])
