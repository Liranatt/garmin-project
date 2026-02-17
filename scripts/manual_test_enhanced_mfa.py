"""
Manual Test for Enhanced Garmin Fetcher MFA
===========================================
This script manually triggers the `_get_mfa_code_from_email` method
in `EnhancedGarminDataFetcher` to verify it can correctly connect to Gmail,
find the Garmin email, and extract the code.

Usage:
    python scripts/manual_test_enhanced_mfa.py
"""

import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.enhanced_fetcher import EnhancedGarminDataFetcher
from dotenv import load_dotenv

# Configure logging to stdout
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("manual_test_mfa")

def main():
    load_dotenv()
    
    log.info("🚀 Starting Manual MFA Check...")
    
    # Credentials are now loaded from .env
    # Check env vars
    email = os.getenv("EMAIL_RECIPIENT") or os.getenv("GARMIN_EMAIL")
    pwd = os.getenv("EMAIL_APP_PASSWORD")
    
    if not email or not pwd:
        log.error("❌ Missing environment variables: EMAIL_APP_PASSWORD and (EMAIL_RECIPIENT or GARMIN_EMAIL)")
        return

    log.info(f"   Using Email User: {email}")
    log.info("   (Password is set)")

    fetcher = EnhancedGarminDataFetcher()
    
    log.info("🔍 Calling _get_mfa_code_from_email()...")
    code = fetcher._get_mfa_code_from_email()
    
    if code:
        log.info(f"✅ SUCCESS! Found MFA Code: {code}")
    else:
        log.warning("⚠️  Result: None (Either no recent email found or connection failed - check logs above)")

if __name__ == "__main__":
    main()
