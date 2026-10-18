from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

from backend.data.editions import (
    borrow_titles,
    group_editions,
    load_editions,
    normalized_title,
    people_key,
    write_editions,
)


CAST_AWAY = "Directors: Robert Zemeckis. Cast: Tom Hanks, Helen Hunt"


def item(item_id, title="", item_doc="", rating_number=None):
    return {"item_id": item_id, "title": title, "item_doc": item_doc, "rating_number": rating_number}


def test_normalized_title_drops_format_labels_but_keeps_seasons():
    assert normalized_title("Cast Away (Two-Disc Special Edition)") == "cast away"
    assert normalized_title("Face/Off [Blu-ray Steelbook]") == "face off"
    assert normalized_title("Aladdin VHS") == "aladdin"
    assert normalized_title("Friends (Season 1)") == "friends season 1"
    assert normalized_title("") == ""


def test_people_key_uses_first_director_and_first_cast_member():
    assert people_key(f"Cast Away. Genres: Drama. {CAST_AWAY}. A man is stranded.") == "robert zemeckis|tom hanks"
    assert people_key("Directors: Edwin L. Marin. Cast: Reginald Owen") == "edwin l. marin|reginald owen"
    assert people_key("Directors: Robert Zemeckis") == ""          # cast missing: too weak to match on
    assert people_key("Directors: ---. Cast: Unavailable") == ""    # placeholder names
    assert people_key("Directors: Various. Cast: Various") == ""


def test_group_editions_needs_shared_title_or_people_and_close_rating_counts():
    items = [
        item("a1", "Cast Away (Two-Disc Special Edition)", f"Cast Away. {CAST_AWAY}", 16753),
        item("a2", "", CAST_AWAY, 16747),                            # Prime Video, no title: linked by people
        item("a3", "Cast Away [Blu-ray]", "Cast Away", 16760),       # linked by title
        item("c1", "A Christmas Carol", "A Christmas Carol", 9900),
        item("c2", "A Christmas Carol [DVD]", "A Christmas Carol", 13700),   # the 1984 remake: other count
        item("s1", "", "Directors: Jack Bender. Cast: Mike Vogel", 20010),
        item("s2", "", "Directors: Jack Bender. Cast: Mike Vogel", 12169),   # another season
        item("n1", "Cast Away", "", None),                           # no rating count: never grouped
        item("h1", "Spider-Man 2", "Directors: Sam Raimi. Cast: Tobey Maguire", 5000),
        item("h2", "", "Directors: Sam Raimi. Cast: Tobey Maguire", 5010),
        item("h3", "Spider-Man 3", "Directors: Sam Raimi. Cast: Tobey Maguire", 5020),  # sequel, close count
        item("p1", "", "Directors: Chris Columbus. Cast: Daniel Radcliffe", 8000),
        item("p2", "Harry Potter: 8-Film Collection [Blu-ray]", "Directors: Chris Columbus. Cast: Daniel Radcliffe", 8010),
    ]

    works = group_editions(items)

    assert works["a1"] == works["a2"] == works["a3"] == "a1"         # work id = smallest item id
    assert works["c1"] != works["c2"]
    assert works["s1"] != works["s2"]
    assert works["n1"] == "n1"
    assert works["h1"] == works["h2"] != works["h3"]                 # titles differ: never one film
    assert works["p1"] != works["p2"]                                # a collection is not an edition


def test_write_and_load_editions(tmp_path):
    rows = [item("a1", "Cast Away", CAST_AWAY, 100), item("a2", "", CAST_AWAY, 101), item("b1", "Heat", "", 5)]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "catalog.parquet")

    stats = write_editions(tmp_path / "catalog.parquet", tmp_path / "editions.parquet")

    assert load_editions(tmp_path / "editions.parquet") == {"a1": "a1", "a2": "a1", "b1": "b1"}
    assert stats == {"items": 3, "films": 2, "items_with_other_editions": 2}


def test_untitled_item_borrows_the_title_of_another_edition(tmp_path):
    rows = [item("a1", "Cast Away [DVD]", CAST_AWAY, 100), item("a2", "", CAST_AWAY, 101), item("b1", "", "", 5)]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "catalog.parquet")
    candidates = {"a2": {"item_id": "a2", "title": ""}, "b1": {"item_id": "b1", "title": ""}}

    borrow_titles(candidates, {"a1": "a1", "a2": "a1", "b1": "b1"}, tmp_path / "catalog.parquet")

    assert candidates["a2"]["title"] == "Cast Away [DVD]"
    assert candidates["a2"]["title_source"] == "meta:a1:title"      # cite the edition the title came from
    assert candidates["b1"]["title"] == ""                          # no other edition: stays untitled


def test_normalized_title_handles_articles_part_numbers_and_years():
    expected = "the twilight saga breaking dawn part 1"

    assert normalized_title("Twilight Saga, The: Breaking Dawn Part 1 [DVD]") == expected
    assert normalized_title("The Twilight Saga: Breaking Dawn Part One") == expected
    assert normalized_title("The Twilight Saga: Breaking Dawn - Part 1 (Two-Disc Special Edition)") == expected
    assert normalized_title("The Twilight Saga: Eclipse (Special Edition) [Blu-ray] (2010)") == "the twilight saga eclipse"


def test_same_title_tolerates_a_wider_rating_gap_than_same_people():
    people = "Directors: David Slade. Cast: Kristen Stewart"
    items = [
        item("t1", "The Twilight Saga: Eclipse (Two-Disc Special Edition)", people, 20129),
        item("t2", "The Twilight Saga: Eclipse (Single-Disc Edition)", people, 20797),      # 3.3% more: same title, one film
        item("p1", "", "Directors: Greg Nicotero. Cast: Norman Reedus", 10000),
        item("p2", "", "Directors: Greg Nicotero. Cast: Norman Reedus", 10300),              # 3% more, people only: kept apart
    ]

    works = group_editions(items)

    assert works["t1"] == works["t2"]
    assert works["p1"] != works["p2"]
