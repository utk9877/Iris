"""Trip triage: four-score ranking + diversification into shortlists (ARCHITECTURE §7)."""

from __future__ import annotations

from iris.triage.presets import PRESETS, TriagePreset, get_preset
from iris.triage.service import TriageItem, TriageResult, TriageService

__all__ = [
    "PRESETS",
    "TriageItem",
    "TriagePreset",
    "TriageResult",
    "TriageService",
    "get_preset",
]
