"""Retrieval-augmented QA over traceable meeting episodes."""

from src.episodic_memory import search_episodes


def retrieve_relevant_memory(
    query: str,
    top_k: int = 5,
    *,
    meeting_id: str | None = None,
    speaker: str | None = None,
    time_range: tuple[float, float] | None = None,
) -> list[dict]:
    """Retrieve relevant meeting episodes for a question.

    Optional ``meeting_id`` / ``speaker`` / ``time_range`` scope retrieval the
    way a user might ("what did Speaker B say in the first 5 minutes?").
    """
    return search_episodes(
        query=query,
        top_k=top_k,
        meeting_id=meeting_id,
        speaker=speaker,
        time_range=time_range,
    )


def answer_question_with_evidence(
    query: str, retrieved_episodes: list[dict]
) -> dict:
    """Answer a question with evidence, speaker, timestamp, and uncertainty.

    The returned dictionary must include ``answer``, ``evidence``, ``speaker``,
    ``timestamp``, ``confidence``, and ``uncertainty_note``.
    """
    if not retrieved_episodes:
        return {
            "answer": "No relevant meeting memory was found.",
            "evidence": [],
            "speaker": "",
            "timestamp": "",
            "confidence": 0.0,
            "uncertainty_note": "Retrieval returned no episodes.",
            "retrieval_method": "",
            "query": query,
        }

    episode = retrieved_episodes[0]
    evidence = episode.get("evidence", [])
    speakers = ", ".join(episode.get("speakers", []))
    timestamp = f"{episode.get('start_time', 0.0):.3f}-{episode.get('end_time', 0.0):.3f}s"
    return {
        "answer": episode.get("summary", ""),
        "evidence": evidence,
        "evidence_ids": episode.get("evidence_ids", []),
        "speaker": speakers,
        "timestamp": timestamp,
        "confidence": episode.get("confidence", 0.0),
        "uncertainty_note": episode.get("uncertainty_note", ""),
        "retrieval_method": episode.get("retrieval_method", ""),
        "retrieval_score": episode.get("retrieval_score", 0.0),
        "query": query,
    }
