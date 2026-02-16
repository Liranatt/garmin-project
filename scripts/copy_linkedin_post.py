import re
import subprocess
import os

def copy_english_post():
    file_path = r"C:\Users\Liran\.gemini\antigravity\brain\31980a2e-98d9-438e-b271-c7dda1e5b53b\linkedin_post.md"
    
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Extract English post section - Robust Regex
        # Matches "## 🇺🇸 English Post" followed by anything until the first code block
        pattern = r"## 🇺🇸 English Post.*?\n\s*```(.*?)```"
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            text = match.group(1).strip()
            # Copy to clipboard using Windows clip command
            # Using UTF-16 encoding which Windows clip expects for unicode
            subprocess.run("clip", input=text.encode("utf-16"), check=True)
            print("Successfully copied AUTHENTIC English post to clipboard!")
            print("-" * 50)
            print(text[:100] + "...")
            print("-" * 50)
        else:
            print("Could not find English post section in file.")
            print("Debug: Content preview:")
            print(content[:500])
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    copy_english_post()
