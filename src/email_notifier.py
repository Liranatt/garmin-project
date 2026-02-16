"""
Email Notifier — Weekly Health Insights
=========================================
Sends a formatted HTML email with the AI agents' weekly recommendations.
Uses Gmail SMTP with App Password authentication.

Setup:
  1. Go to https://myaccount.google.com/apppasswords
  2. Generate an App Password for "Mail"
  3. Set EMAIL_APP_PASSWORD in .env
"""

from __future__ import annotations

import logging
import os
import re
import smtplib
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("email_notifier")


# ═══════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = os.getenv("EMAIL_SENDER", os.getenv("GARMIN_EMAIL", ""))
SENDER_PASSWORD = os.getenv("EMAIL_APP_PASSWORD", "")
RECIPIENT_EMAIL = os.getenv("EMAIL_RECIPIENT", SENDER_EMAIL)


# ═══════════════════════════════════════════════════════════════
#  HTML Template
# ═══════════════════════════════════════════════════════════════

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0f0f0f;
    color: #e0e0e0;
    margin: 0;
    padding: 20px;
  }}
  .container {{
    max-width: 640px;
    margin: 0 auto;
    background: #1a1a2e;
    border-radius: 16px;
    padding: 32px;
    border: 1px solid #2a2a4a;
  }}
  h1 {{
    color: #00d4aa;
    font-size: 24px;
    margin-top: 0;
    border-bottom: 2px solid #2a2a4a;
    padding-bottom: 12px;
  }}
  h2 {{
    color: #7c83ff;
    font-size: 18px;
    margin-top: 28px;
  }}
  .recommendation {{
    background: #16213e;
    border-left: 4px solid #00d4aa;
    padding: 14px 18px;
    margin: 12px 0;
    border-radius: 0 8px 8px 0;
  }}
  .recommendation .title {{
    color: #00d4aa;
    font-weight: 600;
    font-size: 15px;
    margin-bottom: 6px;
  }}
  .recommendation .target {{
    color: #888;
    font-size: 13px;
  }}
  .section {{
    margin: 20px 0;
    padding: 16px;
    background: #16213e;
    border-radius: 8px;
  }}
  .section pre {{
    white-space: pre-wrap;
    word-wrap: break-word;
    font-family: 'Fira Code', 'Consolas', monospace;
    font-size: 13px;
    color: #c0c0c0;
    margin: 0;
  }}
  .footer {{
    text-align: center;
    color: #555;
    font-size: 12px;
    margin-top: 24px;
    padding-top: 16px;
    border-top: 1px solid #2a2a4a;
  }}
  .badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
  }}
  .badge-improve {{ background: #0a3622; color: #00d4aa; }}
  .badge-decline {{ background: #3a1616; color: #ff6b6b; }}
  .badge-stable {{ background: #2a2a16; color: #ffd93d; }}
</style>
</head>
<body>
<div class="container">
  <h1>🏃 Weekly Health Intelligence — {week_date}</h1>
  {recommendations_html}
  {summary_html}
  <div class="footer">
    Garmin Health Intelligence • Generated {generation_date}<br>
    <em>All findings are based on pre-computed statistical analysis, not AI opinions.</em>
  </div>
</div>
</body>
</html>
"""


# ═══════════════════════════════════════════════════════════════
#  Parsing helpers
# ═══════════════════════════════════════════════════════════════

def _extract_recommendations_html(insights: str) -> str:
    """Extract RECOMMENDATION blocks and format as HTML cards."""
    recs = []
    current = None

    for line in insights.splitlines():
        stripped = line.strip()
        upper = stripped.upper()

        if upper.startswith("RECOMMENDATION:"):
            if current and current.get("text"):
                recs.append(current)
            current = {
                "text": stripped.split(":", 1)[1].strip(),
                "target": "",
                "direction": "",
            }
        elif upper.startswith("TARGET_METRIC:") and current is not None:
            current["target"] = stripped.split(":", 1)[1].strip()
        elif upper.startswith("EXPECTED_DIRECTION:") and current is not None:
            current["direction"] = stripped.split(":", 1)[1].strip().upper()

    if current and current.get("text"):
        recs.append(current)

    if not recs:
        return "<p><em>No structured recommendations this week.</em></p>"

    html_parts = ['<h2>🎯 This Week\'s Recommendations</h2>']
    for i, rec in enumerate(recs, 1):
        badge_class = {
            "IMPROVE": "badge-improve",
            "DECLINE": "badge-decline",
            "STABLE": "badge-stable",
        }.get(rec["direction"], "badge-stable")

        direction_text = rec["direction"] if rec["direction"] else "—"
        target_text = rec["target"] if rec["target"] else "general"

        html_parts.append(f"""
        <div class="recommendation">
          <div class="title">#{i}. {rec['text']}</div>
          <div class="target">
            Target: <strong>{target_text}</strong> •
            Expected: <span class="badge {badge_class}">{direction_text}</span>
          </div>
        </div>
        """)

    return "\n".join(html_parts)


def _extract_summary_html(insights: str) -> str:
    """Extract agent section summaries for the email body.

    Shows only the Synthesizer section (bottleneck + quick wins),
    keeping the email concise and actionable.
    """
    # Find the Synthesizer section
    synth_marker = "BOTTLENECK"
    idx = insights.upper().find(synth_marker)
    if idx < 0:
        # Fallback: show last 2000 chars
        snippet = insights[-2000:] if len(insights) > 2000 else insights
    else:
        # Go back to find the section header
        header_start = insights.rfind("=" * 10, 0, idx)
        if header_start < 0:
            header_start = max(0, idx - 200)
        snippet = insights[header_start:]

    # Truncate if extremely long
    if len(snippet) > 4000:
        snippet = snippet[:4000] + "\n\n... [truncated — see full report in dashboard]"

    # Clean up for HTML
    snippet = snippet.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    return f"""
    <h2>📊 Synthesizer Analysis</h2>
    <div class="section">
      <pre>{snippet}</pre>
    </div>
    """


# ═══════════════════════════════════════════════════════════════
#  Main send function
# ═══════════════════════════════════════════════════════════════

def send_weekly_email(insights: str, week_date: date = None) -> bool:
    """Send the weekly health insights email.

    Parameters
    ----------
    insights : str
        Combined text output from all agents.
    week_date : date, optional
        The week this report covers. Defaults to current week start.

    Returns
    -------
    bool
        True if email sent successfully, False otherwise.
    """
    if not SENDER_PASSWORD:
        log.warning("EMAIL_APP_PASSWORD not set in .env — skipping email notification.")
        log.info("To enable email: set EMAIL_APP_PASSWORD to a Gmail App Password.")
        return False

    if not SENDER_EMAIL:
        log.warning("No sender email configured — skipping email notification.")
        return False

    if week_date is None:
        today = date.today()
        week_date = today - __import__("datetime").timedelta(days=today.weekday())

    # Build HTML
    recs_html = _extract_recommendations_html(insights)
    summary_html = _extract_summary_html(insights)

    html_body = HTML_TEMPLATE.format(
        week_date=week_date.strftime("%B %d, %Y"),
        recommendations_html=recs_html,
        summary_html=summary_html,
        generation_date=date.today().strftime("%Y-%m-%d"),
    )

    # Build email
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"🏃 Weekly Health Report — {week_date.strftime('%b %d')}"
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECIPIENT_EMAIL

    # Plain-text fallback (just the raw insights, truncated)
    plain_text = insights[:3000] if len(insights) > 3000 else insights
    msg.attach(MIMEText(plain_text, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    # Send
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, [RECIPIENT_EMAIL], msg.as_string())

        log.info("📧 Weekly email sent to %s", RECIPIENT_EMAIL)
        return True

    except smtplib.SMTPAuthenticationError:
        log.error(
            "Email authentication failed. Make sure EMAIL_APP_PASSWORD is a "
            "valid Gmail App Password (not your regular password).\n"
            "Generate one at: https://myaccount.google.com/apppasswords"
        )
        return False
    except Exception as e:
        log.error("Failed to send email: %s", e)
        return False
