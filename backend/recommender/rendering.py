from __future__ import annotations

from collections.abc import Iterable

from backend.recommender.models import (
    DecisionCertificate,
    EvidenceRecord,
    Explanation,
)


def render_template(
    item_id: str,
    certificate: DecisionCertificate | None,
    evidence: Iterable[EvidenceRecord],
) -> Explanation:
    records = tuple(evidence)
    if certificate is None or not certificate.dual_valid:
        return Explanation(
            item_id,
            "abstained",
            "",
            (),
            "no_supported_dual_valid_coalition_within_budget",
        )

    by_id = {record.citation_id: record for record in records}
    source_citations = tuple(
        source_id
        for source_id in certificate.coalition_source_ids
        if source_id in by_id
        and by_id[source_id].kind == "history"
        and by_id[source_id].source_id == source_id
    )
    certificate_ids = tuple(
        record.citation_id for record in records if record.kind == "certificate"
    )
    if len(source_citations) != certificate.source_count or not certificate_ids:
        return Explanation(
            item_id,
            "abstained",
            "",
            (),
            "incomplete_evidence_bundle",
        )

    citations = source_citations + tuple(
        record.citation_id
        for record in records
        if record.citation_id not in source_citations
    )
    joined_sources = ", ".join(source_citations)
    text = (
        f"{item_id} is supported by history source(s) {joined_sources}. "
        f"Full-pipeline removal moved it outside top-K, while retention kept it "
        f"inside top-K. "
        + "".join(f"[{citation}]" for citation in citations)
    )
    explanation = Explanation(item_id, "verified", text, citations, None)

    if verify_explanation(explanation, certificate, records):
        return explanation

    return Explanation(
        item_id,
        "abstained",
        "",
        (),
        "verification_failed",
    )


def verify_explanation(
    explanation: Explanation,
    certificate: DecisionCertificate | None,
    evidence: Iterable[EvidenceRecord],
) -> bool:
    if certificate is None or not certificate.evaluable or not certificate.dual_valid:
        return False

    if explanation.item_id != certificate.target_item_id:
        return False

    records = {record.citation_id: record for record in evidence}
    if any(citation not in records for citation in explanation.citations):
        return False

    cited_sources = {
        records[citation].source_id
        for citation in explanation.citations
        if records[citation].kind == "history"
    }
    if cited_sources != set(certificate.coalition_source_ids):
        return False

    return any(
        records[citation].kind == "certificate"
        for citation in explanation.citations
    )

