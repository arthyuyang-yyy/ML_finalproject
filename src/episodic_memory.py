"""Create, store, and search traceable meeting episodes.

An *episode* is one coherent unit of meeting memory. Each episode is grouped
around a meeting event (see :func:`src.llm.event_extractor.extract_meeting_events`)
so that the memory layer reasons over events rather than collapsing the whole
meeting into a single blob. Every episode keeps its supporting evidence packets
(``evidence`` / ``evidence_ids``) and aggregated ``uncertainty_note`` so any
retrieved answer can be traced back to the segments that justify it.

Retrieval (:func:`search_episodes`) uses semantic embeddings when
``sentence-transformers`` is installed and falls back to a dependency-free
lexical scorer otherwise, mirroring the optional-heavy-dependency pattern used
elsewhere in the pipeline (SciPy resampling, ASR backends). Both paths support
``meeting_id`` / ``speaker`` / ``time_range`` filters so retrieval can be scoped
the way a real assistant would.
"""

import json
import os
import re
from pathlib import Path
from typing import Any

DEFAULT_MEMORY_PATH = Path("outputs") / "episodic_memory.jsonl"

# Episodes whose evidence segments are separated by more than this many seconds
# are treated as distinct episodes when no explicit event grouping is provided.
DEFAULT_TIME_GAP = 30.0

# Retrieval scoring weights. Relevance dominates; episode importance and recency
# nudge the ranking, and high speech overlap is penalized so uncertain episodes
# do not outrank clean evidence. When semantic embeddings are available the
# relevance term splits across semantic and lexical signals; otherwise the full
# relevance weight falls on the lexical signal.
W_SEMANTIC = 0.45
W_LEXICAL = 0.25
W_IMPORTANCE = 0.15
W_RECENCY = 0.10
W_OVERLAP_PENALTY = 0.20

# Cached embedding model; ``False`` marks "tried and unavailable" so we only pay
# the import/load cost once per process.
_EMBEDDER: Any = None


def create_episode_from_segments(
    segments: list[dict],
    *,
    episode_id: str | None = None,
    topic: str | None = None,
    event_type: str | None = None,
    importance: float | None = None,
) -> dict[str, Any]:
    """Create one coherent episode record from related metadata segments.

    The episode aggregates the evidence packets, their combined confidence, and
    every preserved ``uncertainty_note`` so downstream QA stays traceable. Pass
    ``episode_id`` / ``topic`` / ``event_type`` / ``importance`` to inherit them
    from an upstream meeting event; otherwise they are derived from the segments
    themselves. The episode also records a mean ``overlap_score`` so retrieval
    can down-weight uncertain, high-overlap memories.
    """
    if not segments:
        raise ValueError("segments must not be empty")

    ordered = sorted(segments, key=lambda segment: float(segment["start_time"]))
    meeting_id = str(ordered[0]["meeting_id"])
    speakers = sorted({str(segment.get("speaker", "UNKNOWN")) for segment in ordered})
    evidence_ids = [
        str(segment.get("evidence_id", segment.get("segment_id"))) for segment in ordered
    ]
    text = " ".join(str(segment.get("text", "")).strip() for segment in ordered).strip()
    confidence_values = [
        float(segment.get("asr_confidence", 0.0)) * float(segment.get("speaker_confidence", 0.0))
        for segment in ordered
    ]
    confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
    confidence = round(max(0.0, min(1.0, confidence)), 3)
    overlap_scores = [float(segment.get("overlap_score", 0.0)) for segment in ordered]
    overlap_score = round(sum(overlap_scores) / len(overlap_scores), 3) if overlap_scores else 0.0
    summary = text or "No transcript available."
    return {
        "meeting_id": meeting_id,
        "episode_id": episode_id or f"{meeting_id}_event_0001",
        "event_type": event_type or "discussion",
        "start_time": min(float(segment["start_time"]) for segment in ordered),
        "end_time": max(float(segment["end_time"]) for segment in ordered),
        "speakers": speakers,
        "topic": topic or _derive_topic(summary),
        "summary": summary,
        "evidence_ids": evidence_ids,
        "evidence": ordered,
        "confidence": confidence,
        "overlap_score": overlap_score,
        # Default importance proxies evidence reliability until an LLM assigns a
        # semantic importance; an explicit value (e.g. from an event) overrides it.
        "importance": round(_clamp_unit(importance), 3) if importance is not None else confidence,
        "uncertainty_note": "; ".join(
            str(segment.get("uncertainty_note", ""))
            for segment in ordered
            if segment.get("uncertainty_note")
        ),
    }


def create_episodes_from_segments(
    segments: list[dict],
    events: list[dict] | None = None,
) -> list[dict[str, Any]]:
    """Build one episode per meeting event, preserving the evidence trail.

    When ``events`` (typically from
    :func:`src.llm.event_extractor.extract_meeting_events`) are supplied, each
    event becomes an episode by gathering the segments it references through
    ``evidence_ids``. Segments not covered by any event fall back to time-gap
    grouping so no evidence is silently dropped. With no events at all, the whole
    set is grouped purely by time gaps.
    """
    if not segments:
        return []

    if not events:
        return _episodes_by_time_gap(segments)

    segment_by_evidence = {
        str(segment.get("evidence_id", segment.get("segment_id"))): segment
        for segment in segments
    }
    episodes: list[dict[str, Any]] = []
    covered: set[int] = set()
    for event in events:
        grouped = [
            segment_by_evidence[evidence_id]
            for evidence_id in (str(eid) for eid in event.get("evidence_ids", []))
            if evidence_id in segment_by_evidence
        ]
        if not grouped:
            continue
        covered.update(id(segment) for segment in grouped)
        episodes.append(
            create_episode_from_segments(
                grouped,
                episode_id=event.get("event_id"),
                topic=event.get("summary"),
                event_type=event.get("event_type"),
                importance=event.get("importance"),
            )
        )

    leftover = [segment for segment in segments if id(segment) not in covered]
    episodes.extend(_episodes_by_time_gap(leftover, start_index=len(episodes) + 1))
    return episodes


def store_episode(episode: dict, path: str | Path = DEFAULT_MEMORY_PATH) -> None:
    """Persist a single episode by appending one JSONL record."""
    store_episodes([episode], path)


def store_episodes(episodes: list[dict], path: str | Path = DEFAULT_MEMORY_PATH) -> None:
    """Persist multiple episodes as JSONL, creating the parent directory."""
    memory_path = Path(path)
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    with memory_path.open("a", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode, ensure_ascii=False) + "\n")


def search_episodes(
    query: str,
    top_k: int = 5,
    path: str | Path = DEFAULT_MEMORY_PATH,
    *,
    meeting_id: str | None = None,
    speaker: str | None = None,
    time_range: tuple[float, float] | None = None,
) -> list[dict]:
    """Return the most relevant episodes for a query.

    Episodes are first filtered by ``meeting_id`` / ``speaker`` / ``time_range``
    (any of which may be ``None`` to skip that filter), then ranked. Ranking uses
    semantic similarity when an embedding backend is available and a lexical
    overlap score otherwise. Each returned episode carries ``retrieval_score`` and
    ``retrieval_method`` so the ranking stays auditable.
    """
    memory_path = Path(path)
    if not memory_path.exists():
        return []

    candidates = [
        episode
        for episode in _load_episodes(memory_path)
        if _passes_filters(episode, meeting_id, speaker, time_range)
    ]
    if not candidates:
        return []

    scores, method = _score_candidates(query, candidates)

    ranked = sorted(
        (
            {**episode, "retrieval_score": round(float(score), 4), "retrieval_method": method}
            for episode, score in zip(candidates, scores)
        ),
        key=lambda episode: episode["retrieval_score"],
        reverse=True,
    )
    matched = [episode for episode in ranked if episode["retrieval_score"] > 0]
    return (matched or ranked)[:top_k]


def _episodes_by_time_gap(
    segments: list[dict],
    gap: float = DEFAULT_TIME_GAP,
    start_index: int = 1,
) -> list[dict[str, Any]]:
    """Group time-ordered segments into episodes whenever a long pause occurs."""
    if not segments:
        return []

    ordered = sorted(segments, key=lambda segment: float(segment["start_time"]))
    groups: list[list[dict]] = [[ordered[0]]]
    for segment in ordered[1:]:
        previous_end = float(groups[-1][-1]["end_time"])
        if float(segment["start_time"]) - previous_end > gap:
            groups.append([segment])
        else:
            groups[-1].append(segment)

    meeting_id = str(ordered[0]["meeting_id"])
    return [
        create_episode_from_segments(group, episode_id=f"{meeting_id}_event_{start_index + offset:04d}")
        for offset, group in enumerate(groups)
    ]


def _passes_filters(
    episode: dict,
    meeting_id: str | None,
    speaker: str | None,
    time_range: tuple[float, float] | None,
) -> bool:
    """Return whether an episode satisfies the optional retrieval filters."""
    if meeting_id is not None and str(episode.get("meeting_id")) != str(meeting_id):
        return False
    if speaker is not None and str(speaker) not in episode.get("speakers", []):
        return False
    if time_range is not None:
        window_start, window_end = time_range
        if float(episode.get("end_time", 0.0)) < window_start:
            return False
        if float(episode.get("start_time", 0.0)) > window_end:
            return False
    return True


def _searchable_text(episode: dict) -> str:
    """Concatenate the fields used for ranking an episode."""
    return " ".join(
        [
            str(episode.get("summary", "")),
            str(episode.get("topic", "")),
            " ".join(episode.get("speakers", [])),
        ]
    )


def _score_candidates(query: str, candidates: list[dict]) -> tuple[list[float], str]:
    """Blend relevance, importance, recency, and an overlap penalty per episode.

    ``final = relevance + W_IMPORTANCE * importance + W_RECENCY * recency
    - W_OVERLAP_PENALTY * overlap_score``. Relevance combines semantic and
    lexical signals when embeddings are available, otherwise the full relevance
    weight falls on the lexical signal. Returns the scores and the relevance
    method used ("semantic" or "lexical").
    """
    documents = [_searchable_text(episode) for episode in candidates]
    semantic = _semantic_similarities(query, documents)
    lexical = [_lexical_score(query, document) for document in documents]
    recency = _recency_scores(candidates)
    method = "semantic" if semantic is not None else "lexical"

    scores: list[float] = []
    for index, episode in enumerate(candidates):
        if semantic is not None:
            relevance = W_SEMANTIC * semantic[index] + W_LEXICAL * lexical[index]
        else:
            relevance = (W_SEMANTIC + W_LEXICAL) * lexical[index]
        final = (
            relevance
            + W_IMPORTANCE * _clamp_unit(episode.get("importance", episode.get("confidence", 0.0)))
            + W_RECENCY * recency[index]
            - W_OVERLAP_PENALTY * _clamp_unit(episode.get("overlap_score", 0.0))
        )
        scores.append(max(0.0, final))
    return scores, method


def _semantic_similarities(query: str, documents: list[str]) -> list[float] | None:
    """Return per-document cosine similarities in [0, 1], or None if unavailable."""
    embedder = _get_embedder()
    if embedder is None:
        return None
    import numpy as np

    vectors = embedder.encode([query, *documents], normalize_embeddings=True)
    query_vector = np.asarray(vectors[0])
    document_vectors = np.asarray(vectors[1:])
    similarities = document_vectors @ query_vector
    # Map cosine similarity from [-1, 1] into [0, 1] for a stable score.
    return [float((value + 1.0) / 2.0) for value in similarities]


def _recency_scores(candidates: list[dict]) -> list[float]:
    """Score episodes by how late they end, normalized to [0, 1] within the set.

    Within a single meeting this proxies recency by position on the timeline;
    when all episodes share an end time (or there is only one), recency is
    neutral so it never dominates the ranking.
    """
    end_times = [float(episode.get("end_time", 0.0)) for episode in candidates]
    earliest, latest = min(end_times), max(end_times)
    if latest == earliest:
        return [0.0 for _ in candidates]
    span = latest - earliest
    return [(end_time - earliest) / span for end_time in end_times]


def _lexical_score(query: str, document: str) -> float:
    """Return the share of query terms present in the document (CJK-aware)."""
    query_terms = _terms(query)
    if not query_terms:
        return 0.0
    document_terms = _terms(document)
    overlap = len(query_terms & document_terms)
    return overlap / len(query_terms)


def _terms(text: str) -> set[str]:
    """Tokenize into lowercase words plus individual CJK characters.

    Whitespace tokenization alone drops Chinese, which is unsegmented; emitting
    each CJK character as its own term gives the lexical fallback meaningful
    overlap on the Chinese fixtures while still handling English words.
    """
    lowered = text.lower()
    words = {token for token in re.findall(r"[a-z0-9]+", lowered) if token}
    cjk = set(re.findall(r"[一-鿿]", lowered))
    return words | cjk


def _derive_topic(summary: str, max_length: int = 40) -> str:
    """Derive a short topic label from a summary when none is supplied."""
    collapsed = " ".join(summary.split())
    if len(collapsed) <= max_length:
        return collapsed
    return collapsed[:max_length].rstrip() + "…"


def _clamp_unit(value: Any) -> float:
    """Clamp a score-like value into ``[0.0, 1.0]``."""
    return max(0.0, min(1.0, float(value)))


def _load_episodes(memory_path: Path) -> list[dict]:
    """Read all stored episodes from a JSONL memory file."""
    episodes: list[dict] = []
    for line in memory_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            episodes.append(json.loads(line))
    return episodes


def _get_embedder() -> Any:
    """Load the optional sentence-transformers backend once, if available.

    Returns the model, or ``None`` when the dependency is missing or disabled
    via ``EPISODIC_DISABLE_SEMANTIC``. The model name can be overridden with
    ``EPISODIC_EMBED_MODEL``.
    """
    global _EMBEDDER
    if _EMBEDDER is None:
        if os.environ.get("EPISODIC_DISABLE_SEMANTIC"):
            _EMBEDDER = False
        else:
            try:
                from sentence_transformers import SentenceTransformer

                model_name = os.environ.get("EPISODIC_EMBED_MODEL", "all-MiniLM-L6-v2")
                _EMBEDDER = SentenceTransformer(model_name)
            except Exception:
                _EMBEDDER = False
    return _EMBEDDER or None
