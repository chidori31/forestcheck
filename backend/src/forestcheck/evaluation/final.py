"""Stage-10 release gate. Training/dev entry points continue to forbid holdout."""

import json
from datetime import datetime, timezone
from pathlib import Path


def verify_release(frozen: dict, current: dict) -> None:
    if frozen.get("schema_version") != "1.0" or frozen != current:
        raise ValueError("final_release_changed")


def claim_final(directory: Path, digest: str) -> None:
    """Exclusive marker is deliberately retained on failed or interrupted evaluation."""
    with (directory / "consumed.json").open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "freeze_sha256": digest,
                "exposed_at": datetime.now(timezone.utc).isoformat(),
                "meaning": "reserved internal holdout released; never tune on this cohort",
            },
            stream,
        )
