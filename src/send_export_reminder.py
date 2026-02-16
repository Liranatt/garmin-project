"""
Send Export Reminder
====================
Sends an email to the user reminding them to manually trigger the Garmin data export.
This should be scheduled (e.g., weekly) to ensure the bulk import pipeline has fresh data.
"""

import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.email_notifier import send_generic_email, load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("export_reminder")

SUBJECT = "📥 Action Required: Request Garmin Data Export"

HTML_BODY = """
<!DOCTYPE html>
<html>
<body style="font-family: sans-serif; background: #1a1a2e; color: #e0e0e0; padding: 20px;">
<div style="max-width: 600px; margin: 0 auto; background: #16213e; padding: 30px; border-radius: 10px; border: 1px solid #2a2a4a;">
  <h2 style="color: #00d4aa; margin-top: 0;">Time to Update Your Data! ⏱️</h2>
  
  <p>To keep your deep analytics (workouts, GPS, etc.) up to date, please trigger a new export.</p>
  
  <div style="text-align: center; margin: 30px 0;">
    <a href="https://www.garmin.com/en-US/account/datamanagement/exportdata/" 
       style="background: #00d4aa; color: #000; padding: 15px 30px; text-decoration: none; font-weight: bold; border-radius: 5px; display: inline-block;">
       Request Data Export
    </a>
  </div>

  <p style="font-size: 14px; color: #aaa;">
    <strong>What happens next?</strong><br>
    1. You click the button above.<br>
    2. Garmin prepares the file.<br>
    3. You'll get an email from Garmin.<br>
    4. Our system detects that email and auto-processes it (if scheduled).
  </p>
  
  <hr style="border: 0; border-top: 1px solid #2a2a4a; margin: 20px 0;">
  <p style="font-size: 12px; color: #555;">Garmin Health Intelligence Automation</p>
</div>
</body>
</html>
"""

def main():
    log.info("Sending export reminder...")
    if send_generic_email(SUBJECT, HTML_BODY):
        log.info("Reminder sent successfully.")
    else:
        log.error("Failed to send reminder.")
        sys.exit(1)

if __name__ == "__main__":
    main()
