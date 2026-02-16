import re
import subprocess

def copy_english_post():
    file_path = r"C:\Users\Liran\.gemini\antigravity\brain\31980a2e-98d9-438e-b271-c7dda1e5b53b\linkedin_post.md"
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Extract English post section
        pattern = r"## 🇺🇸 English Post\s+```(.*?)```"
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            text = match.group(1).strip()
            # Copy to clipboard using Windows clip command
            subprocess.run("clip", input=text.encode("utf-16"), check=True)
            print("Successfully copied English post to clipboard!")
        else:
            print("Could not find English post section in file.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    copy_english_post()
