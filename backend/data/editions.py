"""Group catalog items that are the same film (DVD, Blu-ray, VHS and Prime Video editions).

One film is often sold under several item ids. Without grouping, a user who watched the
Prime Video "Cast Away" is recommended the DVD "Cast Away", and a recommendation of the DVD
counts as a miss when the target is the Blu-ray.

Two items are editions of the same film when both hold:
1. they share a normalised title ("Cast Away [Blu-ray]" -> "cast away") or the same first
   director + first cast member (Prime Video items often have no title, only people);
2. their rating counts differ by at most 2%: Amazon shows the ratings of all formats of a film
   together, so editions have nearly the same count (Cast Away: 16,747 / 16,753 / 16,756).
   Remakes and the seasons of a series have clearly different counts and stay apart.
Guards against chaining: two groups that both have titles must share one ("Spider-Man 2" never
joins "Spider-Man 3"), placeholder names ("Various") are ignored, and bundles ("8-Film
Collection", "Complete Series", box sets) are never grouped with anything.

Output: editions.parquet with (item_id, work_id); work_id is the smallest item id of the film.
"""

from __future__ import annotations

import re

from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


RATING_TOLERANCE = 0.02

# Bracketed labels like "(Two-Disc Special Edition)" or "[Blu-ray]"; "(Season 1)" is kept.
FORMAT_LABEL = re.compile(
    r"blu-?ray|dvd|vhs|edition|disc|widescreen|full ?screen|digital|\bhd\b|4k|uhd|combo|region"
    r"|steelbook|import|anniversary|remaster|restored|collector|special|deluxe|unrated|cut\b"
    r"|extended|theatrical|subtitled|ntsc|\bpal\b|exclusive",
    re.IGNORECASE,
)
BRACKETS = re.compile(r"[\[(]([^\])]*)[\])]")
FORMAT_WORD = re.compile(r"\b(blu-?ray|dvd|vhs)\b", re.IGNORECASE)

# Bundles hold several films or seasons, so they are never an edition of one film.
BUNDLE = re.compile(
    r"\b(collection|trilogy|quadrilogy|box ?set|boxset|complete|anthology|double feature|\d+[- ](film|movie)s?|pack)\b",
    re.IGNORECASE,
)

FIRST_DIRECTOR = re.compile(r"Directors: (.+?)(?:, |\. Cast: |$)")
FIRST_CAST = re.compile(r"Cast: (.+?)(?:, |\. |$)")

# Placeholder names shared by thousands of unrelated items ("Directors: Various. Cast: Various").
NOT_A_NAME = {"various", "none", "unavailable", "unknown", "n/a", "na", "not available", "various artists"}


def normalized_title(title: str) -> str:
    title = BRACKETS.sub(lambda match: " " if FORMAT_LABEL.search(match.group(1)) else match.group(0), title or "")
    title = FORMAT_WORD.sub(" ", title)

    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def people_key(item_doc: str) -> str:
    """'first director|first cast member', or '' when either is missing or a placeholder."""
    director = FIRST_DIRECTOR.search(item_doc or "")
    cast = FIRST_CAST.search(item_doc or "")

    if not director or not cast:
        return ""

    names = [director.group(1).strip().lower(), cast.group(1).strip().lower()]
    if any(not re.search(r"[a-z]", name) or name in NOT_A_NAME for name in names):
        return ""

    return "|".join(names)


def group_editions(items: list[dict[str, Any]]) -> dict[str, str]:
    """item_id -> work_id for every item (an item without other editions is its own work)."""
    parent = {item["item_id"]: item["item_id"] for item in items}
    titles = {item["item_id"]: {normalized_title(item["title"])} - {""} for item in items}

    def root(item_id: str) -> str:
        while parent[item_id] != item_id:
            item_id = parent[item_id]

        return item_id

    def join(a: str, b: str) -> None:
        a, b = root(a), root(b)

        # Two groups that both have titles must share one: an untitled Prime Video item with the
        # same director and star must not chain "Spider-Man 2" to "Spider-Man 3".
        if a == b or (titles[a] and titles[b] and not titles[a] & titles[b]):
            return

        keep, drop = min(a, b), max(a, b)                 # the smallest id stays the root
        parent[drop] = keep
        titles[keep] |= titles.pop(drop)

    # Items that share a title or the same people are candidates for being one film.
    shared: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for item in items:
        if not item["rating_number"] or BUNDLE.search(item["title"] or ""):
            continue

        title, people = normalized_title(item["title"]), people_key(item["item_doc"])
        if title:
            shared[f"title:{title}"].append(item)
        if people:
            shared[f"people:{people}"].append(item)

    # Within one candidate group, items whose rating counts are within 2% of each other are one film.
    for group in shared.values():
        group.sort(key=lambda item: (item["rating_number"], item["item_id"]))
        first = group[0]

        for item in group[1:]:
            if item["rating_number"] <= first["rating_number"] * (1 + RATING_TOLERANCE):
                join(first["item_id"], item["item_id"])
            else:
                first = item

    return {item_id: root(item_id) for item_id in parent}


def write_editions(catalog_path: Path, output_path: Path) -> dict[str, Any]:
    items = pq.read_table(catalog_path, columns=["item_id", "title", "item_doc", "rating_number"]).to_pylist()
    works = group_editions(items)

    item_ids = sorted(works)
    pq.write_table(pa.table({"item_id": item_ids, "work_id": [works[i] for i in item_ids]}), output_path)

    sizes: dict[str, int] = defaultdict(int)
    for work_id in works.values():
        sizes[work_id] += 1

    return {
        "items": len(works),
        "films": len(sizes),
        "items_with_other_editions": sum(size for size in sizes.values() if size > 1),
    }


def borrow_titles(rows: dict[str, dict[str, Any]], editions: dict[str, str], catalog_path: Path) -> None:
    """Give untitled catalog rows (many Prime Video items) the title of another edition of the film.

    `title_source` keeps the id of the edition the title came from, so an explanation cites it correctly.
    """
    films = {editions.get(item_id, item_id) for item_id, row in rows.items() if not row.get("title")}
    others = sorted(item_id for item_id, film in editions.items() if film in films and item_id not in rows)

    if not others:
        return

    table = pq.read_table(catalog_path, columns=["item_id", "title"], filters=[("item_id", "in", others)])
    titles: dict[str, tuple[str, str]] = {}

    for other in table.to_pylist():
        if other["title"]:
            titles.setdefault(editions[other["item_id"]], (other["title"], other["item_id"]))

    for item_id, row in rows.items():
        film = editions.get(item_id, item_id)

        if not row.get("title") and film in titles:
            row["title"], source = titles[film]
            row["title_source"] = f"meta:{source}:title"


def load_editions(path: Path) -> dict[str, str]:
    table = pq.read_table(path)

    return dict(zip(table["item_id"].to_pylist(), table["work_id"].to_pylist()))
