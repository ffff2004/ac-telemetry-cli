"""The explicit, producer-assembled persisted telemetry dataset contract."""

from .assist_activity import ACTIVITY_TABLE_SPECS
from .contract_types import DatasetContract
from .events import EVENT_TABLE_SPECS
from .replay import REPLAY_TABLE_SPECS
from .segments import SEGMENT_TABLE_SPECS
from .setup_parser import SETUP_TABLE_SPECS
from .summary import SUMMARY_TABLE_SPECS
from .track import TRACK_TABLE_SPECS

DATASET_CONTRACT = DatasetContract(
    tables=(
        *REPLAY_TABLE_SPECS,
        *TRACK_TABLE_SPECS,
        *SETUP_TABLE_SPECS,
        *EVENT_TABLE_SPECS,
        *ACTIVITY_TABLE_SPECS,
        *SEGMENT_TABLE_SPECS,
        *SUMMARY_TABLE_SPECS,
    )
)
DATASET_CONTRACT.validate_definition()
