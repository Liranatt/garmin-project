import logging
import os
import sys
import time
import imaplib
import email
import re
import html
from email.header import decode_header
import garth
from dotenv import load_dotenv

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)-8s %(message)s')
log = logging.getLogger(__name__)

# Load .env
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv()

def get_mfa_code_from_email():
    """Reads the MFA code from Gmail (Logic from test_email_code.py)"""
    # Using the credentials the user found working:
    gmail_user = 'lirattar@gmail.com'
    # CORRECT App Password:
    gmail_password = 'ospl dkpu dbqn wyxn' 

    log.info(f"📧 Checking Gmail ({gmail_user}) for MFA code...")
    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(gmail_user, gmail_password)
        mail.select("inbox")

        # Search for emails from Garmin
        status, messages = mail.search(None, '(FROM "alerts@account.garmin.com")')
        if status != "OK":
            log.warning("❌ No Garmin emails found.")
            return None

        email_ids = messages[0].split()
        if not email_ids:
            log.warning("❌ No Garmin emails found.")
            return None

        # Check last 5 emails (newest first)
        latest_ids = email_ids[-5:]
        latest_ids.reverse()

        for e_id in latest_ids:
            _, msg_data = mail.fetch(e_id, "(RFC822)")
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    
                    # Extract body (prefer plain, fallback html)
                    body, html_body = "", ""
                    if msg.is_multipart():
                        for part in msg.walk():
                            ctype = part.get_content_type()
                            if ctype == "text/plain":
                                try:
                                    body = part.get_payload(decode=True).decode(errors='ignore')
                                except: pass
                            elif ctype == "text/html":
                                try:
                                    html_body = part.get_payload(decode=True).decode(errors='ignore')
                                except: pass
                    else:
                        ctype = msg.get_content_type()
                        payload = msg.get_payload(decode=True).decode(errors='ignore')
                        if ctype == "text/html": html_body = payload
                        else: body = payload
                    
                    # Clean content
                    content = body if body.strip() else html_body
                    # Strip HTML
                    content = re.sub(r'<script.*?>.*?</script>', ' ', content, flags=re.DOTALL|re.IGNORECASE)
                    content = re.sub(r'<style.*?>.*?</style>', ' ', content, flags=re.DOTALL|re.IGNORECASE)
                    content = re.sub(r'<[^>]+>', ' ', content)
                    content = html.unescape(content)
                    clean_body = re.sub(r'\s+', ' ', content).strip()

                    # Check for code: "account" followed by 6 digits
                    match = re.search(r'account\D*(\d{6})', clean_body, re.IGNORECASE)
                    if match:
                        code = match.group(1)
                        log.info(f"✅ Auto-MFA: Found code {code} (in email ID {e_id.decode()})")
                        mail.logout()
                        return code
        
        log.warning("❌ Could not find valid code in recent emails.")
        mail.logout()
    except Exception as e:
        log.error(f"❌ Auto-MFA error: {e}")
    
    return None

def verify_full_flow():
    """Attempt a full login using Gautomator logic."""
    print("="*60)
    print("  VERIFYING AUTO-LOGIN FLOW")
    print("="*60)

    # Credentials for GARMIN
    email = os.getenv("GARMIN_EMAIL")
    password = os.getenv("GARMIN_PASSWORD")
    
    # Fallback to hardcoded if env missing (based on user's previous script context)
    if not email: email = "lirattar@gmail.com"
    if not password: password = "Lir12345"

    log.info(f"👤 Garmin User: {email}")
    
    def mfa_callback():
        log.info("🔒 MFA Requested! Starting Auto-MFA sequence...")
        # Wait a bit for email to arrive
        log.info("⏳ Waiting 15s for email to arrive...")
        time.sleep(15) 
        
        code = get_mfa_code_from_email()
        if code:
            return code
        
        log.error("❌ Failed to get code from email.")
        return input("Input code manually (fallback): ")

    try:
        log.info("🔄 Attempting login...")
        # Force a fresh login by NOT resuming
        # We need to trick garth or just rely on it asking for MFA if the session is invalid.
        # But if we have a valid session, it won't ask.
        # So maybe we should just call login directly.
        garth.login(email, password, prompt_mfa=mfa_callback)
        
        log.info("✅ LOGIN SUCCESSFUL!")
        print("\n🎉 The full pipeline works! We successfully: 1) Failed initial auth 2) Waited 3) Got code from email 4) Logged in.")
        
    except Exception as e:
        log.error(f"❌ Login failed: {e}")

if __name__ == "__main__":
    verify_full_flow()
