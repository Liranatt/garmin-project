"""
Tests for the email notifier module.

Covers:
- _extract_recommendations_html parsing
- _extract_summary_html extraction
- send_weekly_email HTML assembly
- send_generic_email guard clauses
"""
import sys
import os
from datetime import date
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from email_notifier import (
    _extract_recommendations_html,
    _extract_summary_html,
    send_generic_email,
    send_weekly_email,
)


# ─── _extract_recommendations_html ──────────────────────────


class TestExtractRecommendationsHtml:

    def test_parses_structured_recommendations(self):
        text = (
            "RECOMMENDATION: Sleep 7+ hours on training days\n"
            "TARGET_METRIC: sleep_score\n"
            "EXPECTED_DIRECTION: IMPROVE\n"
            "\n"
            "RECOMMENDATION: Reduce alcohol on rest days\n"
            "TARGET_METRIC: hrv_last_night\n"
            "EXPECTED_DIRECTION: IMPROVE\n"
        )
        html = _extract_recommendations_html(text)
        assert "Sleep 7+ hours" in html
        assert "Reduce alcohol" in html
        assert "sleep_score" in html
        assert "badge-improve" in html

    def test_handles_decline_direction(self):
        text = (
            "RECOMMENDATION: Lower resting HR through zone-2 training\n"
            "TARGET_METRIC: resting_hr\n"
            "EXPECTED_DIRECTION: DECLINE\n"
        )
        html = _extract_recommendations_html(text)
        assert "badge-decline" in html

    def test_handles_stable_direction(self):
        text = (
            "RECOMMENDATION: Maintain current sleep schedule\n"
            "TARGET_METRIC: sleep_score\n"
            "EXPECTED_DIRECTION: STABLE\n"
        )
        html = _extract_recommendations_html(text)
        assert "badge-stable" in html

    def test_no_recommendations_returns_fallback(self):
        html = _extract_recommendations_html("Just some random text with no structured data.")
        assert "No structured recommendations" in html

    def test_empty_string(self):
        html = _extract_recommendations_html("")
        assert "No structured recommendations" in html

    def test_recommendation_without_target(self):
        text = "RECOMMENDATION: Take a rest day\n"
        html = _extract_recommendations_html(text)
        assert "Take a rest day" in html
        assert "general" in html  # fallback target

    def test_multiple_recommendations_numbered(self):
        text = (
            "RECOMMENDATION: First rec\n"
            "TARGET_METRIC: metric_a\n"
            "EXPECTED_DIRECTION: IMPROVE\n"
            "\n"
            "RECOMMENDATION: Second rec\n"
            "TARGET_METRIC: metric_b\n"
            "EXPECTED_DIRECTION: IMPROVE\n"
            "\n"
            "RECOMMENDATION: Third rec\n"
            "TARGET_METRIC: metric_c\n"
            "EXPECTED_DIRECTION: DECLINE\n"
        )
        html = _extract_recommendations_html(text)
        assert "#1." in html
        assert "#2." in html
        assert "#3." in html

    def test_case_insensitive_parsing(self):
        text = (
            "recommendation: Lower case rec\n"
            "target_metric: stress_level\n"
            "expected_direction: improve\n"
        )
        html = _extract_recommendations_html(text)
        assert "Lower case rec" in html


# ─── _extract_summary_html ───────────────────────────────────


class TestExtractSummaryHtml:

    def test_extracts_synthesizer_section(self):
        text = (
            "============ PATTERNS ============\n"
            "Some pattern stuff\n"
            "============ BOTTLENECK & QUICK WINS ============\n"
            "The #1 bottleneck is insufficient recovery.\n"
            "Quick wins: sleep more, train less.\n"
        )
        html = _extract_summary_html(text)
        assert "BOTTLENECK" in html
        assert "insufficient recovery" in html

    def test_fallback_on_no_synthesizer(self):
        text = "No structured sections here. Just raw output without bottleneck."
        html = _extract_summary_html(text)
        # Should use the last 2000 chars as fallback
        assert "No structured sections" in html

    def test_truncation_of_very_long_input(self):
        text = "BOTTLENECK\n" + "x" * 5000
        html = _extract_summary_html(text)
        # Should be truncated
        assert "truncated" in html

    def test_html_entities_escaped(self):
        text = "BOTTLENECK\nValue <script>alert('xss')</script> should be safe"
        html = _extract_summary_html(text)
        assert "<script>" not in html
        assert "&lt;script&gt;" in html


# ─── send_generic_email ──────────────────────────────────────


class TestSendGenericEmail:

    def test_skips_when_no_password(self):
        """SENDER_PASSWORD is set at import time, so we patch the module attribute."""
        import email_notifier
        original = email_notifier.SENDER_PASSWORD
        email_notifier.SENDER_PASSWORD = ""
        try:
            result = send_generic_email("Subject", "<p>Body</p>")
            assert result is False
        finally:
            email_notifier.SENDER_PASSWORD = original

    @patch("email_notifier.smtplib.SMTP")
    @patch.dict(os.environ, {
        "EMAIL_APP_PASSWORD": "fake_pass",
        "GARMIN_EMAIL": "sender@test.com",
        "EMAIL_RECIPIENT": "recipient@test.com",
    })
    def test_sends_email_successfully(self, mock_smtp_class):
        """Test successful email send with mocked SMTP."""
        mock_server = MagicMock()
        mock_smtp_class.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_class.return_value.__exit__ = MagicMock(return_value=False)

        # Override module-level constants for this test
        import email_notifier
        original_sender = email_notifier.SENDER_EMAIL
        original_password = email_notifier.SENDER_PASSWORD
        email_notifier.SENDER_EMAIL = "sender@test.com"
        email_notifier.SENDER_PASSWORD = "fake_pass"

        try:
            result = send_generic_email(
                "Test Subject",
                "<p>Test body</p>",
                recipient="recipient@test.com",
            )
            assert result is True
        finally:
            email_notifier.SENDER_EMAIL = original_sender
            email_notifier.SENDER_PASSWORD = original_password


# ─── send_weekly_email ───────────────────────────────────────


class TestSendWeeklyEmail:

    @patch("email_notifier.send_generic_email", return_value=True)
    def test_formats_subject_with_date(self, mock_send):
        insights = (
            "RECOMMENDATION: Rest day tomorrow\n"
            "TARGET_METRIC: training_readiness\n"
            "EXPECTED_DIRECTION: IMPROVE\n"
        )
        result = send_weekly_email(insights, week_date=date(2026, 2, 24))
        assert result is True
        args = mock_send.call_args
        subject = args[0][0]
        assert "Feb 24" in subject

    @patch("email_notifier.send_generic_email", return_value=True)
    def test_html_contains_recommendations(self, mock_send):
        insights = (
            "RECOMMENDATION: Increase deep sleep\n"
            "TARGET_METRIC: deep_sleep_sec\n"
            "EXPECTED_DIRECTION: IMPROVE\n"
        )
        send_weekly_email(insights, week_date=date(2026, 2, 24))
        args = mock_send.call_args
        html_body = args[0][1]
        assert "Increase deep sleep" in html_body
        assert "deep_sleep_sec" in html_body

    @patch("email_notifier.send_generic_email", return_value=True)
    def test_plain_text_fallback(self, mock_send):
        insights = "x" * 5000
        send_weekly_email(insights, week_date=date(2026, 2, 24))
        args = mock_send.call_args
        plain_text = args[1].get("plain_text") or args[0][2]
        # Should be truncated to 3000 chars
        assert len(plain_text) <= 3000
