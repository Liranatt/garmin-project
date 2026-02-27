"""
Tests for the summary builder module.

Covers: build_concise_summary edge cases and content extraction.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline.summary_builder import build_concise_summary


class TestBuildConciseSummary:

    def test_empty_input_returns_defaults(self):
        result = build_concise_summary("")
        assert "What changed" in result
        assert "Why it matters" in result
        assert "Next 24-48h" in result
        assert "Insufficient data" in result

    def test_none_input_returns_defaults(self):
        result = build_concise_summary(None)
        assert "Insufficient data" in result

    def test_extracts_three_bullets(self):
        result = build_concise_summary(
            "Sleep score improved by 10% this week.\n"
            "This matters because recovery capacity is tied to deep sleep.\n"
            "Recommendation: keep training moderate and focus on sleep timing."
        )
        lines = result.strip().split("\n")
        assert len(lines) == 3
        assert lines[0].startswith("- What changed:")
        assert lines[1].startswith("- Why it matters:")
        assert lines[2].startswith("- Next 24-48h:")

    def test_keyword_based_extraction(self):
        result = build_concise_summary(
            "Preamble text that should be skipped.\n"
            "HRV has shown an increasing trend over the last week.\n"
            "This impacts recovery quality and readiness.\n"
            "You should reduce intensity tomorrow.\n"
        )
        assert "trend" in result.lower() or "increasing" in result.lower()
        assert "recovery" in result.lower() or "impacts" in result.lower()
        assert "reduce" in result.lower() or "should" in result.lower()

    def test_skips_table_rows(self):
        """Markdown table rows (|...|) should be filtered out."""
        result = build_concise_summary(
            "| Metric | Value |\n"
            "| resting_hr | 54 |\n"
            "HRV showed an upward trend.\n"
            "This means recovery is improving.\n"
            "Recommendation: maintain current training load.\n"
        )
        assert "Metric" not in result
        assert "resting_hr" not in result

    def test_skips_separator_lines(self):
        result = build_concise_summary(
            "=================\n"
            "Sleep score declined this week.\n"
            "Because of poor REM sleep quality.\n"
            "Focus on consistent bedtime tonight.\n"
        )
        assert "===" not in result

    def test_truncation_of_long_lines(self):
        long_line = "Sleep quality improved significantly " + "x" * 500
        result = build_concise_summary(
            f"{long_line}\n"
            "Recovery matters more than volume.\n"
            "Next step: moderate pace tomorrow.\n"
        )
        # Each bullet should be clipped
        for line in result.split("\n"):
            assert len(line) <= 280

    def test_fallback_when_no_keywords_match(self):
        """When no keywords match, use first available lines."""
        result = build_concise_summary(
            "First line of raw output.\n"
            "Second line of raw output.\n"
            "Third line of raw output.\n"
        )
        lines = result.strip().split("\n")
        assert len(lines) == 3
        assert "First line" in lines[0]
        assert "Second line" in lines[1]
        assert "Third line" in lines[2]

    def test_real_world_input(self):
        """Test with a snippet similar to actual agent output."""
        real_input = (
            "============================================================\n"
            "  BOTTLENECK & QUICK WINS (Synthesizer)\n"
            "============================================================\n"
            "\n"
            "Sleep score declined by 8.5% week-over-week.\n"
            "This matters because sustained poor sleep means recovery capacity drops.\n"
            "RECOMMENDATION: Reduce training intensity for the next 3 days.\n"
            "TARGET_METRIC: training_readiness\n"
            "EXPECTED_DIRECTION: IMPROVE\n"
        )
        result = build_concise_summary(real_input)
        lines = result.strip().split("\n")
        assert len(lines) == 3
        # Should pick up the decline as "what changed"
        assert "declined" in lines[0].lower() or "sleep" in lines[0].lower()
