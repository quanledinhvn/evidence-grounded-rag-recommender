"""Markdown report of an explanation run: data, every step of every sampled query, overall numbers.

Written by `python -m backend.generation.experiment ... --report docs/results/<name>.md` (Vietnamese,
for the course report). The last section ("Đánh giá tổng quan") is left for a person to write.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.data.editions import borrow_titles
from backend.eval.evaluate import film_rank
from backend.generation.sources import item_name, item_sources
from backend.generation.validator import kept, render
from backend.profile.profile import load_catalog
from backend.service import EXPLAINED, FUSION, QUERIES


def day(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%".replace(".", ",")


def write_report(path: Path, args: Any, drawn: dict, records: list[dict], explains: dict, summary: dict, fusion: Any, editions: dict[str, str]) -> None:
    """Every step of every sampled query, then the overall numbers (Vietnamese, for the course report)."""
    lines = [
        f"# Smoke test {len(records)} record — hệ thống cuối (fusion + bằng chứng + giải thích có trích dẫn)",
        "",
        f"Sinh tự động bởi `python -m backend.generation.experiment` ({datetime.now().strftime('%Y-%m-%d %H:%M')}). "
        "Phần \"Đánh giá tổng quan\" ở cuối do người chạy viết.",
        "",
        "## 1. Dữ liệu và cấu hình",
        "",
        "| Thành phần | Giá trị |",
        "|---|---|",
        f"| Pha | `{args.phase}`: model fit trên train" + (" + validation; target là tương tác cuối của user (test)" if args.phase == "test" else "; target là tương tác validation") + " |",
        f"| Query | `{args.artifacts_dir / QUERIES[args.phase]}` — mỗi user một target, `as_of` = thời điểm target |",
        f"| Mẫu | {len(drawn['random'])} query ngẫu nhiên (`Random({args.seed})`) + {len(drawn['with_target'])} query có target trong top 10 "
        f"(lấy đầu tiên trong {drawn['scanned']} query ngẫu nhiên; tỉ lệ có target trong top 10 của lượt quét: {pct(drawn['hit_rate_at_10'])}) |",
        f"| Lịch sử, hồ sơ | `splits.parquet`, chỉ tương tác trước `as_of`; catalog `catalog_enriched.parquet`; phiên bản `editions.parquet` |",
        f"| Xếp hạng | Fusion `{args.model_dir or FUSION[args.phase]}`: trọng số " + ", ".join(f"{k} {v:g}" for k, v in fusion.config["weights"].items())
        + f", method {fusion.config['method']}, depth {fusion.config['depth']}, gom phiên bản; giữ nguyên thứ tự fusion (không xếp lại) |",
        "| Bằng chứng | Metadata catalog + tối đa 2 review của người khác viết trước `as_of` (`reviews.sqlite`) cho mỗi phim top 10 |",
        f"| LLM giải thích | `{args.model}`, temperature 0, JSON schema; top {EXPLAINED}, tối đa 2 câu/phim; 2 chế độ: tự do và bắt buộc trích dẫn |",
        f"| Validator | ID nguồn tồn tại và đúng phim, H# thuộc hồ sơ, tên người/tựa trong câu có trong nguồn, độ giống MiniLM ≥ {args.min_similarity:g} |",
        f"| Giám khảo | `{args.judge_model}` (thinking low), chấm từng câu: supported / partial / unsupported so với nguồn của phim và lịch sử |",
        "",
        "## 2. Chi tiết từng record",
        "",
    ]
    names = target_names(records, args.artifacts_dir / "catalog_enriched.parquet", editions)

    for n, record in enumerate(records, 1):
        query, package, profile = record["query"], record["package"], record["profile"]
        user = query["user_id"]
        top_ids = [item["item_id"] for item in record["ranked"]]
        rank = film_rank(top_ids, query["target_item_id"], editions)
        explained = explains[user]
        group = "ngẫu nhiên" if record["group"] == "random" else "có target trong top 10"

        lines += [
            f"### Record {n} ({group}) — user `{user}`",
            "",
            f"- **Query:** `as_of` = {query['as_of']} ({day(query['as_of'])}); target `{query['target_item_id']}` "
            f"— *{names[query['target_item_id']]}*, rating {query['target_rating']:.0f}★.",
            f"- **Kết quả:** target " + (f"**ở hạng {rank}** (tính theo cùng phim)" if rank else "không có trong top 10") + ".",
            "",
            f"**Bước 1 — Hồ sơ.** {profile['seen']} phim đã xem trước `as_of`; {profile['positive']} phim thích (≥ 4★), "
            f"{profile['negative']} phim không thích (≤ 2★); {profile['without_metadata']} phim không có metadata. Bằng chứng lịch sử đưa cho LLM:",
            "",
        ]
        for hid, entry in list(package["history"].items())[:6]:
            own = f' — review của user: "{short(entry["own_review"], 20)}"' if entry.get("own_review") else ""
            lines.append(f"- [{hid}] {entry['rating']:.0f}★ {short(entry['text'], 14)}{own}")
        if len(package["history"]) > 6:
            lines.append(f"- … và {len(package['history']) - 6} mục khác")

        lines += ["", "**Bước 2 — Fusion top 10** (nhánh tìm ra phim, điểm fusion):", "", "| Hạng | Phim | Nhánh | Điểm |", "|---:|---|---|---:|"]
        for k, item in enumerate(record["ranked"], 1):
            mark = " ✅ target" if film_rank([item["item_id"]], query["target_item_id"], editions) else ""
            lines.append(f"| {k} | {short(item_name(package, item['item_id']), 12)}{mark} | {'+'.join(item.get('members', [item['source']]))} | {item['score']:.3f} |")

        evidence_counts = Counter(entry["source_type"] for entry in package["evidence"].values())
        lines += ["", f"**Bước 3 — Bằng chứng.** {len(package['evidence'])} mục cho top 10 ({evidence_counts['metadata']} metadata, "
                      f"{evidence_counts['review']} review). Của top {EXPLAINED}:", ""]
        for item_id in explained["items"]:
            lines.append(f"- *{short(item_name(package, item_id), 12)}*: " + "; ".join(
                f"[{eid}] {short(entry['text'], 12)}" for eid, entry in item_sources(package, item_id)))

        usage = explained["cited"]
        lines += ["", f"**Bước 4 — LLM giải thích có trích dẫn** ({usage['input_tokens']} token vào, {usage['output_tokens']} token ra"
                      f"{', cache' if usage['cache_hit'] else ''}; trạng thái {usage['status']}) và **bước 5 — validator, giám khảo**:", "",
                  "| Phim | Câu LLM viết | Nguồn | Validator | Độ giống | Giám khảo |", "|---|---|---|---|---:|---|"]
        for item_id, claims in explained["cited"]["items"].items():
            for claim in claims or [{"text": "(không có câu nào)", "evidence_ids": [], "history_ids": [], "problems": [], "similarity": 0.0}]:
                verdict = "giữ" if claim.get("text") != "(không có câu nào)" and kept(claim, args.min_similarity) else (
                    ", ".join(claim["problems"]) or ("độ giống thấp" if claim.get("text") != "(không có câu nào)" else "—"))
                lines.append(f"| {short(item_name(package, item_id), 6)} | {claim['text']} | {' '.join(claim['evidence_ids'] + claim['history_ids'])} | "
                             f"{verdict} | {claim['similarity']:.2f} | {claim.get('label', '—')} |")

        lines += ["", "**Lời giải thích hiển thị cho user** (chỉ các câu validator giữ lại):", ""]
        for item_id, claims in explained["cited"]["items"].items():
            valid = [c for c in claims if kept(c, args.min_similarity)]
            lines.append(f"- *{short(item_name(package, item_id), 12)}*: " + (render(valid) if valid else "(không có giải thích — evidence_unavailable)"))

        lines += ["", "**Đối chứng — giải thích tự do** (không bắt buộc trích dẫn, cùng thông tin):", ""]
        for item_id, claims in explained["free"]["items"].items():
            lines.append(f"- *{short(item_name(package, item_id), 12)}*: " + " ".join(f"{c['text']} *({c.get('label', '—')})*" for c in claims))
        lines.append("")

    lines += overall_section(summary)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def overall_section(summary: dict) -> list[str]:
    g = summary["G"]
    lines = ["## 3. Số liệu tổng hợp", "", "| Nhóm | Số user | NDCG@10 | HR@10 |", "|---|---:|---:|---:|"]
    for group, values in summary["ranking"].items():
        lines.append(f"| {group} | {values['users']} | {pct(values['ndcg_at_10'])} | {pct(values['hr_at_10'])} |")

    lines += ["", f"Giải thích ({g['items_requested']} phim cần giải thích):", "",
              "| Cấu hình | Số câu | Phim có giải thích | supported | partial | unsupported |", "|---|---:|---:|---:|---:|---:|"]
    for name, label in (("free", "Tự do"), ("cited_raw", "Bắt buộc trích dẫn, trước validator"), ("cited_validated", "Bắt buộc trích dẫn, sau validator")):
        values = g[name]
        labels = values["labels"]
        lines.append(f"| {label} | {values['claims']} | {values['items_explained']} | {labels.get('supported', 0)} | {labels.get('partial', 0)} | {labels.get('unsupported', 0)} |")

    removed = g["removed_by_validator"]
    lines += ["", f"- Validator loại {removed['claims']} câu: nhãn giám khảo {removed['labels']}; lý do: {g['problems'] or 'chỉ độ giống thấp'}.",
              "- Theo ngưỡng độ giống: " + "; ".join(f"{t}: {v['claims']} câu, supported {pct(v['supported_rate'])}" for t, v in g["similarity_thresholds"].items()) + ".",
              f"- Lỗi sinh: {g['generation_failed']}.",
              f"- Token: giải thích có trích dẫn {g['usage']['cited']['input_tokens']} vào / {g['usage']['cited']['output_tokens']} ra; "
              f"tự do {g['usage']['free']['input_tokens']} / {g['usage']['free']['output_tokens']}; giám khảo {g['judge_usage']['input_tokens']} / {g['judge_usage']['output_tokens']}.",
              f"- Độ trễ một lời gọi giải thích có trích dẫn (không tính cache): p50 {fmt_ms(g['usage']['cited']['latency_ms_p50'])}, p95 {fmt_ms(g['usage']['cited']['latency_ms_p95'])}.",
              "", "## 4. Đánh giá tổng quan", "", "(viết sau khi đọc kết quả)", ""]

    return lines


def fmt_ms(value: float | None) -> str:
    return "—" if value is None else f"{value / 1000:.1f} s".replace(".", ",")


def short(text: str, words: int) -> str:
    parts = (text or "").replace("|", "/").split()
    return " ".join(parts[:words]) + (" …" if len(parts) > words else "")


def target_names(records: list[dict], catalog_path: Path, editions: dict[str, str]) -> dict[str, str]:
    """Title of every target (borrowed from another edition if needed), else the start of its item_doc."""
    targets = {r["query"]["target_item_id"] for r in records}
    rows = load_catalog(catalog_path, targets)
    borrow_titles(rows, editions, catalog_path)

    return {t: short(rows.get(t, {}).get("title") or rows.get(t, {}).get("item_doc") or "(không có metadata)", 12) for t in targets}
