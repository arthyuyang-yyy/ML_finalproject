"""Deterministic, non-LLM structured event extraction baselines."""

from .rule_extractor import extract_rule_based_events

__all__ = ["extract_rule_based_events"]
