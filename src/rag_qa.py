"""Compatibility facade for retrieval-augmented meeting QA."""

from src.qa import answer_question_with_evidence, retrieve_relevant_memory


__all__ = ["answer_question_with_evidence", "retrieve_relevant_memory"]
