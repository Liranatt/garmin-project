"""Helpers for building concise insight text for UI consumption."""

from __future__ import annotations


def build_concise_summary(insights: str) -> str:
    """Create a strict 3-bullet, human-friendly summary for UI cards."""
    text = str(insights or "").replace("\r", "").strip()
    if not text:
        return (
            "- What changed: Insufficient data in this run.\n"
            "- Why it matters: Without stable signal, training decisions should stay conservative.\n"
            "- Next 24-48h: Keep effort moderate and reassess after tomorrow's sync."
        )

    candidates = []
    for raw in text.splitlines():
        line = raw.strip().lstrip("-* ").strip()
        if not line:
            continue
        if line.startswith("|") and line.endswith("|"):
            continue
        if set(line) <= {"=", "-", ":"}:
            continue
        candidates.append(line)

    def clip(s: str, limit: int = 260) -> str:
        s = s.replace("\n", " ").strip()
        if len(s) <= limit:
            return s
        return s[: limit - 3].rstrip() + "..."

    def bullet(label: str, value: str) -> str:
        prefix = f"- {label}: "
        allowed = max(48, 280 - len(prefix))
        return prefix + clip(value, allowed)

    what_changed = ""
    why_it_matters = ""
    next_24_48h = ""

    for line in candidates:
        low = line.lower()
        if not what_changed and any(
            t in low
            for t in (
                "trend",
                "up",
                "down",
                "increase",
                "decrease",
                "improved",
                "declined",
                "drop",
                "rise",
                "stable",
            )
        ):
            what_changed = line
            continue
        if not why_it_matters and any(
            t in low
            for t in (
                "because",
                "means",
                "risk",
                "impact",
                "matters",
                "recovery",
                "fatigue",
                "stress",
                "readiness",
            )
        ):
            why_it_matters = line
            continue
        if not next_24_48h and any(
            t in low
            for t in ("recommend", "should", "next", "today", "tomorrow", "focus", "avoid", "keep", "do")
        ):
            next_24_48h = line

    for line in candidates:
        if not what_changed:
            what_changed = line
            continue
        if not why_it_matters and line != what_changed:
            why_it_matters = line
            continue
        if not next_24_48h and line not in (what_changed, why_it_matters):
            next_24_48h = line

    what_changed = clip(what_changed or "Week-over-week signal changed, but data density is limited.")
    why_it_matters = clip(why_it_matters or "This affects recovery quality and your expected response to training load.")
    next_24_48h = clip(next_24_48h or "Keep intensity moderate, prioritize sleep timing, and reassess after the next daily run.")

    return (
        f"{bullet('What changed', what_changed)}\n"
        f"{bullet('Why it matters', why_it_matters)}\n"
        f"{bullet('Next 24-48h', next_24_48h)}"
    )

