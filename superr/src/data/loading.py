"""Raw table loading. No cleaning happens here; see ``cleaning.py``."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.config import resolve_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RawTables:
    """The six CSVs of the data pack, exactly as supplied."""

    balls: pd.DataFrame
    matches: pd.DataFrame
    competitions: pd.DataFrame
    teams: pd.DataFrame
    playing_eleven: pd.DataFrame
    player_mapping: pd.DataFrame


def load_raw_tables(config: dict[str, Any]) -> RawTables:
    """Load every CSV named in ``config['files']`` from ``config['paths']['data_dir']``."""
    data_dir = resolve_path(config["paths"]["data_dir"])
    files = config["files"]
    frames = {}
    for key, filename in files.items():
        frames[key] = pd.read_csv(data_dir / filename)
        logger.info("Loaded %s: %d rows x %d cols", filename, *frames[key].shape)
    return RawTables(
        balls=frames["ball_by_ball"],
        matches=frames["matches"],
        competitions=frames["competitions"],
        teams=frames["teams"],
        playing_eleven=frames["playing_eleven"],
        player_mapping=frames["player_mapping"],
    )
