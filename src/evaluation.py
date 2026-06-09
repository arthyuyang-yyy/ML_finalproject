"""Evaluation interfaces for recognition, memory, evidence, and uncertainty."""


def evaluate_overlap_routing(predictions: list[str], references: list[str]) -> dict:
    """Evaluate low/high-overlap routing predictions."""
    # TODO: compute accuracy, precision, recall, and F1.
    raise NotImplementedError("Overlap routing evaluation is not implemented yet.")


def evaluate_evidence_support(predictions: list[dict], references: list[dict]) -> dict:
    """Evaluate evidence hit rate, unsupported claims, and hallucination."""
    # TODO: define matching rules for timestamped evidence and supported claims.
    raise NotImplementedError("Evidence evaluation is not implemented yet.")
