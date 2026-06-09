"""Retrieval-augmented QA over traceable meeting episodes."""

from src.episodic_memory import search_episodes


def retrieve_relevant_memory(query: str, top_k: int = 5) -> list[dict]:
    """Retrieve relevant meeting episodes for a question."""
    return search_episodes(query=query, top_k=top_k)


def answer_question_with_evidence(
    query: str, retrieved_episodes: list[dict]
) -> dict:
    """Answer a question with evidence, speaker, timestamp, and uncertainty.

    The returned dictionary must include ``answer``, ``evidence``, ``speaker``,
    ``timestamp``, ``confidence``, and ``uncertainty_note``.
    """
    # TODO: constrain the QA model to retrieved evidence and validate citations.
    raise NotImplementedError("Evidence-backed QA is not implemented yet.")
