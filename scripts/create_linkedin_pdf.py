from PIL import Image
import os

def create_pdf():
    image_paths = [
        "docs/screenshots/overview.png",
        "docs/screenshots/correlations.png",
        "docs/screenshots/agent_chat.png"
    ]
    
    # Verify images exist
    images = []
    for path in image_paths:
        if os.path.exists(path):
            img = Image.open(path).convert('RGB')
            images.append(img)
            print(f"Loaded {path}")
        else:
            print(f"Warning: {path} not found")
            
    if images:
        output_path = "docs/project_showcase.pdf"
        images[0].save(
            output_path, 
            save_all=True, 
            append_images=images[1:]
        )
        print(f"Created PDF carousel at {output_path}")
    else:
        print("No images found to create PDF")

if __name__ == "__main__":
    create_pdf()
