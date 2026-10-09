from __future__ import annotations

import math
from datetime import datetime
from datetime import timezone

from typing import Any


TEXT_FIELD_GROUPS = (
    ("title",),
    ("description",),
    ("feature", "features"),
    ("category", "categories"),
)


def is_valid_id(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def has_usable_text(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())

    if not isinstance(value, list):
        return False

    return any(isinstance(part, str) and bool(part.strip()) for part in value)


def usable_text_count(record: dict[str, Any]) -> int:
    return sum(
        any(has_usable_text(record.get(field)) for field in fields)
        for fields in TEXT_FIELD_GROUPS
    )


def review_user_id(record: dict[str, Any]) -> Any:
    return record.get("reviewerID", record.get("user_id"))


def review_item_id(record: dict[str, Any]) -> Any:
    parent_asin = record.get("parent_asin")
    if is_valid_id(parent_asin):
        return parent_asin

    return record.get("asin")


def review_rating(record: dict[str, Any]) -> Any:
    return record.get("overall", record.get("rating"))


def review_timestamp(record: dict[str, Any]) -> Any:
    value = record.get("unixReviewTime", record.get("timestamp"))
    if isinstance(value, int) and not isinstance(value, bool) and value >= 100_000_000_000:
        return value // 1000

    return value


def metadata_item_id(record: dict[str, Any]) -> Any:
    parent_asin = record.get("parent_asin")
    if is_valid_id(parent_asin):
        return parent_asin

    return record.get("asin")


def review_issues(record: dict[str, Any], audit_time: datetime) -> set[str]:
    issues: set[str] = set()

    if not is_valid_id(review_user_id(record)):
        issues.add("user_id")

    if not is_valid_id(review_item_id(record)):
        issues.add("item_id")

    rating = review_rating(record)
    rating_valid = (
        isinstance(rating, (int, float))
        and not isinstance(rating, bool)
        and math.isfinite(rating)
        and 1 <= rating <= 5
    )
    if not rating_valid:
        issues.add("rating")

    timestamp = review_timestamp(record)
    audit_timestamp = _as_utc(audit_time).timestamp()
    timestamp_valid = (
        isinstance(timestamp, int)
        and not isinstance(timestamp, bool)
        and 0 < timestamp <= audit_timestamp
    )

    if not timestamp_valid:
        issues.add("timestamp")

    return issues


def review_date_mismatch(record: dict[str, Any]) -> bool:
    if "reviewTime" not in record:
        return False

    review_time = record.get("reviewTime")
    timestamp = review_timestamp(record)
    if not isinstance(review_time, str) or not review_time.strip():
        return True

    if not isinstance(timestamp, int) or isinstance(timestamp, bool) or timestamp <= 0:
        return False

    try:
        text_date = datetime.strptime(review_time.strip(), "%m %d, %Y").date()
        timestamp_date = datetime.fromtimestamp(timestamp, timezone.utc).date()
    except (OverflowError, OSError, ValueError):
        return True

    return text_date != timestamp_date


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc)
