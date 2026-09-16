"""Case-set helpers shared by the standing-row scripts (vendored from the lab repository).

load_cases() returns the standing pool (341 cases across six languages) from ../cases; mode_of() labels a case by
how its root cause is stated (existing / found / nominal), for the segment breakdowns; primary_rel() picks the main
causal relation from an extractor's output. The floor-threshold sweep that gave this file its lab name is not
ported here -- it depends on a private span-linker artifact.
"""
from standing_row import load_cases, read_cases  # noqa: F401  (re-exported for the other scripts)

__all__ = ["load_cases", "read_cases", "mode_of", "primary_rel"]


def mode_of(case):
    """existing = original standing case; found/nominal = the r11 extension, split by how the root is phrased."""
    if not str(case["id"]).startswith("r11-"):
        return "existing"
    return "nominal" if "nominal-root" in case.get("notes", "") else "found"


def primary_rel(relations):
    """The main cause/effect relation: the one with the most span text."""
    return max(relations, key=lambda r: len(r.get("cause", "")) + len(r.get("effect", ""))) if relations else None
