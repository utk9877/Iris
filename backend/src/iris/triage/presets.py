"""Triage preset profiles (ARCHITECTURE §7).

Each preset weights the four normalized scores — ``quality`` (technical), ``aesthetic``,
``representativeness`` (cosine to the scope centroid), ``subject`` (face presence ×
quality × known-person bonus) — and sets the MMR trade-off ``lam`` (higher = favour
the score over diversity). ``delete-candidates`` is inverted: it ranks by *badness*
(redundancy + low quality) and skips MMR, surfacing all-but-best of each near-dup /
burst group. Weights are starting values, tuned empirically (never fabricated).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TriagePreset:
    """A named triage profile: score weights + diversification behaviour."""

    name: str
    description: str
    weight_quality: float
    weight_aesthetic: float
    weight_repr: float
    weight_subject: float
    lam: float  # MMR: score weight vs. diversity (1 - lam)
    invert: bool = False  # True -> surface delete-candidates (worst/redundant first)


PRESETS: dict[str, TriagePreset] = {
    "story-ready": TriagePreset(
        name="story-ready",
        description="Balanced, diverse picks that tell the trip's story.",
        weight_quality=0.25,
        weight_aesthetic=0.25,
        weight_repr=0.30,
        weight_subject=0.20,
        lam=0.6,
    ),
    "print-worthy": TriagePreset(
        name="print-worthy",
        description="Favour beauty and technical quality; allow similar shots if both excel.",
        weight_quality=0.35,
        weight_aesthetic=0.45,
        weight_repr=0.10,
        weight_subject=0.10,
        lam=0.85,
    ),
    "delete-candidates": TriagePreset(
        name="delete-candidates",
        description="Redundant and low-quality shots — the all-but-best of each cluster.",
        weight_quality=0.5,
        weight_aesthetic=0.2,
        weight_repr=0.0,
        weight_subject=0.0,
        lam=1.0,
        invert=True,
    ),
}

DEFAULT_PRESET = "story-ready"


def get_preset(name: str | None) -> TriagePreset:
    """Look up a preset by name, falling back to the default. Raises on unknown names."""
    if name is None:
        return PRESETS[DEFAULT_PRESET]
    if name not in PRESETS:
        raise KeyError(name)
    return PRESETS[name]
