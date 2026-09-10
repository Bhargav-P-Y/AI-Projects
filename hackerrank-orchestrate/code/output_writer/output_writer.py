import logging
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd

from llm_router.llm_router import RoutingDecision, ALLOWED_ACTIONS, ALLOWED_MESSAGE_TYPES

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "message_id",
    "action",
    "message_type",
    "reason",
    "confidence",
    "evidence_message_ids",
]


class OutputWriter:
    """Validator and CSV exporter for Message Notification Router output.csv.

    Enforces strict 8-point schema compliance:
    1. Exact row count matching dataset/messages.csv (110 rows).
    2. Exact message_id list and ordering.
    3. Action in {'notify', 'digest', 'mute'}.
    4. Message type in the 11 allowed taxonomy types.
    5. Confidence is float in range [0.0, 1.0].
    6. Reason is a non-empty string.
    7. Evidence message IDs is non-empty ('none' or separated IDs).
    8. Exact column headers and ordering.
    """

    @staticmethod
    def validate_decisions(
        decisions: Dict[str, RoutingDecision],
        expected_messages: pd.DataFrame,
    ) -> List[str]:
        """Validates decisions against expected dataset messages and schema rules.

        Returns a list of error string messages (empty if 100% valid).
        """
        errors: List[str] = []
        expected_ids = expected_messages["message_id"].tolist()

        # 1. Row count validation
        if len(decisions) != len(expected_ids):
            errors.append(
                f"Row count mismatch: decisions has {len(decisions)} items, "
                f"expected {len(expected_ids)} from messages.csv"
            )

        # 2. Per-message schema and order validation
        for idx, expected_id in enumerate(expected_ids):
            if expected_id not in decisions:
                errors.append(f"Missing decision for message_id '{expected_id}' at index {idx}")
                continue

            dec = decisions[expected_id]

            # ID match check
            if dec.message_id != expected_id:
                errors.append(
                    f"ID mismatch at row {idx}: expected '{expected_id}', got '{dec.message_id}'"
                )

            # Action check
            if dec.action not in ALLOWED_ACTIONS:
                errors.append(
                    f"Message '{expected_id}': invalid action '{dec.action}'. Must be one of {ALLOWED_ACTIONS}"
                )

            # Message type check
            if dec.message_type not in ALLOWED_MESSAGE_TYPES:
                errors.append(
                    f"Message '{expected_id}': invalid message_type '{dec.message_type}'. "
                    f"Must be one of {ALLOWED_MESSAGE_TYPES}"
                )

            # Confidence check
            if not isinstance(dec.confidence, (int, float)) or not (0.0 <= dec.confidence <= 1.0):
                errors.append(
                    f"Message '{expected_id}': invalid confidence '{dec.confidence}'. Must be float in [0.0, 1.0]"
                )

            # Reason check
            if not dec.reason or not isinstance(dec.reason, str) or not dec.reason.strip():
                errors.append(f"Message '{expected_id}': reason is empty or missing")

            # Evidence message IDs check
            if dec.evidence_message_ids is None or not str(dec.evidence_message_ids).strip():
                errors.append(
                    f"Message '{expected_id}': evidence_message_ids is empty (use 'none' if no evidence)"
                )

        return errors

    @classmethod
    def to_dataframe(
        cls,
        decisions: Dict[str, RoutingDecision],
        expected_messages: pd.DataFrame,
    ) -> pd.DataFrame:
        """Converts decisions dict into a formatted pandas DataFrame matching expected_messages order."""
        rows = []
        expected_ids = expected_messages["message_id"].tolist()

        for msg_id in expected_ids:
            if msg_id in decisions:
                dec = decisions[msg_id]
                rows.append({
                    "message_id": dec.message_id,
                    "action": dec.action,
                    "message_type": dec.message_type,
                    "reason": dec.reason,
                    "confidence": round(float(dec.confidence), 2),
                    "evidence_message_ids": dec.evidence_message_ids,
                })
            else:
                logger.warning(f"Message ID '{msg_id}' missing from decisions during DataFrame creation.")

        df = pd.DataFrame(rows)
        return df[REQUIRED_COLUMNS]

    @classmethod
    def write_csv(
        cls,
        decisions: Dict[str, RoutingDecision],
        expected_messages: pd.DataFrame,
        output_path: Union[str, Path] = "dataset/output.csv",
    ) -> pd.DataFrame:
        """Validates decisions, builds DataFrame, and writes to output.csv.

        Raises ValueError if validation fails.
        """
        # Validate first
        errors = cls.validate_decisions(decisions, expected_messages)
        if errors:
            err_msg = f"Output CSV validation failed with {len(errors)} error(s):\n" + "\n".join(errors[:10])
            logger.error(err_msg)
            raise ValueError(err_msg)

        # Build DataFrame
        df = cls.to_dataframe(decisions, expected_messages)

        # Write to path with UTF-8 encoding and \n line endings
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False, encoding="utf-8", lineterminator="\n")

        logger.info(f"Successfully wrote {len(df)} validated rows to '{out_path}'.")
        return df
