"""Candidate generation package facade."""

from .generator import generate_candidates, generate_high_overlap_candidates

__all__ = ["generate_candidates", "generate_high_overlap_candidates"]
