import garth
import os
import base64
import json
from getpass import getpass

def main():
    print("="*60)
    print("  GITHUB ACTIONS SECRET GENERATOR")
    print("="*60)
    print("This script will generate the base64 strings you need for your GitHub Secrets.")
    
    email = input("Enter Garmin Email: ")
    password = getpass("Enter Garmin Password: ")
    
    # Ensure ~/.garth exists
    garth_home = os.path.expanduser("~/.garth")
    os.makedirs(garth_home, exist_ok=True)

    print(f"\nLogging in as {email}...")
    try:
        garth.login(email, password)
        print("✅ Login successful!")
    except Exception as e:
        print(f"❌ Login failed: {e}")
        return

    # Save session to disk so we can read the raw JSON files
    garth.save(garth_home)
    
    print("\n" + "="*60)
    print("  SECRETS TO COPY")
    print("="*60)

    # 1. GARTH_SESSION (oauth1)
    oauth1_path = os.path.join(garth_home, "oauth1_token.json")
    if os.path.exists(oauth1_path):
        with open(oauth1_path, "rb") as f:
            raw_bytes = f.read()
            # Verify it's valid JSON
            try:
                json.loads(raw_bytes)
                b64_str = base64.b64encode(raw_bytes).decode("utf-8")
                print(f"\nName: GARTH_SESSION")
                print(f"Value:\n{b64_str}")
            except json.JSONDecodeError:
                 print(f"\n❌ Error: generated oauth1_token.json is not valid JSON.")

    else:
        print("\n❌ Error: oauth1_token.json was not generated.")

    # 2. GARTH_SESSION_OAUTH2 (oauth2)
    oauth2_path = os.path.join(garth_home, "oauth2_token.json")
    if os.path.exists(oauth2_path):
        with open(oauth2_path, "rb") as f:
            raw_bytes = f.read()
             # Verify it's valid JSON
            try:
                json.loads(raw_bytes)
                b64_str = base64.b64encode(raw_bytes).decode("utf-8")
                print(f"\nName: GARTH_SESSION_OAUTH2")
                print(f"Value:\n{b64_str}")
            except json.JSONDecodeError:
                 print(f"\n❌ Error: generated oauth2_token.json is not valid JSON.")
    else:
        print("\n⚠️ Warning: oauth2_token.json was not generated (might not be needed yet, but usually is).")

    print("\n" + "="*60)
    print("INSTRUCTIONS:")
    print("1. Go to your GitHub Repo -> Settings -> Secrets and variables -> Actions")
    print("2. Create/Update 'GARTH_SESSION' with the first value above.")
    print("3. Create/Update 'GARTH_SESSION_OAUTH2' with the second value above.")
    print("="*60)

if __name__ == "__main__":
    main()
