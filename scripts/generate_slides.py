import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
import matplotlib.image as mpimg
from PIL import Image
import os
import math

# Configuration
THEME = {
    'bg': '#0E1117',        # Streamlit Dark
    'text': '#FAFAFA',      # White-ish
    'accent': '#00F5A0',    # Teal
    'secondary': '#667EEA', # Purple
    'dim': '#A0A0A0',       # Grey
}

SLIDE_SIZE = (16, 9)
DPI = 100

def setup_slide(title):
    fig = plt.figure(figsize=SLIDE_SIZE, dpi=DPI, facecolor=THEME['bg'])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(THEME['bg'])
    
    # Title
    ax.text(0.05, 0.90, title.upper(), color=THEME['accent'], 
            fontsize=24, fontweight='bold', fontname='Arial')
    
    # Branding - Bottom Right
    ax.text(0.95, 0.05, "Garmin Health Intelligence // Engineering Deep Dive", 
            color=THEME['dim'], fontsize=10, ha='right', fontname='Consolas')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    return fig, ax

def add_bullet_points(ax, points, x=0.1, y=0.75, spacing=0.08, fontsize=18):
    for pt in points:
        ax.text(x, y, "• " + pt, color=THEME['text'], fontsize=fontsize, va='top', fontname='Arial', wrap=True)
        y -= spacing

def add_code_block(ax, code, x=0.1, y=0.5, fontsize=12):
    ax.text(x, y, code, color='#E0E0E0', fontsize=fontsize, 
            fontname='Consolas', va='top', bbox=dict(facecolor='#1E1E1E', edgecolor='#333333', pad=10))

def add_image(ax, path, x, y, width, height, caption=None):
    if not os.path.exists(path):
        print(f"Warning: Image not found {path}")
        ax.text(x + width/2, y + height/2, f"[Missing Image: {os.path.basename(path)}]", 
                color='red', ha='center', va='center')
        return

    try:
        img = Image.open(path)
        ax.imshow(img, extent=(x, x+width, y, y+height), aspect='auto', zorder=1)
        
        # Border
        rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor=THEME['accent'], facecolor='none', zorder=2)
        ax.add_patch(rect)
        
        if caption:
            ax.text(x + width/2, y - 0.04, caption, color=THEME['dim'], ha='center', fontsize=12, style='italic')
    except Exception as e:
        print(f"Error checking image {path}: {e}")

def create_slides():
    output_pdf = "docs/garmin_engineering_deep_dive.pdf"
    
    with PdfPages(output_pdf) as pdf:
        
        # SLIDE 1: Authentic Title
        fig, ax = setup_slide("Garmin Health Intelligence")
        ax.text(0.5, 0.6, "How do you prevent\nAI from Hallucinating?", 
                color=THEME['text'], fontsize=40, ha='center', fontweight='bold')
        ax.text(0.5, 0.35, "An Engineering approach to Personal Health Analytics\nInspired by Prof. Oren Freifeld's 'Intro to Deep Learning' (BGU)", 
                color=THEME['secondary'], fontsize=16, ha='center', style='italic')
        
        tech_stack = "Python • PostgreSQL • CrewAI • Streamlit • Scipy • Heroku"
        ax.text(0.5, 0.2, tech_stack, color=THEME['dim'], fontsize=14, ha='center', fontname='Consolas')
        pdf.savefig(fig)
        plt.close()

        # SLIDE 2: The Core Philosophy (The Pipe)
        fig, ax = setup_slide("Determinism Before Semantics")
        points = [
            "Problem: LLMs fed raw data will invent correlations.",
            "Solution: Constrain the AI to look through a 'Pipe'.",
            "The Pipe = Validated Statistical Results Only.",
            "The AI can't see raw data, only what the math proves."
        ]
        add_bullet_points(ax, points)
        
        # Visual diagram of "The Pipe"
        diagram = """
        [ Raw Garmin Data ]
               ⬇
        [ DETERMINISTIC MATH LAYER ] (The Guardrail)
        • Pearson Correlations (Linear)
        • Markov Chains (State Transitions)
        • AR(1) Models (Persistence)
               ⬇
        [ SEMANTIC LAYER ] (The AI)
        • 5 Specialized Agents (Read-Only)
        • "Here is the math. Explain it."
               ⬇
        [ True Insight ]
        """
        add_code_block(ax, diagram, x=0.55, y=0.8, fontsize=10)
        pdf.savefig(fig)
        plt.close()

        # SLIDE 3: The Dashboard (Evidence)
        fig, ax = setup_slide("The Result: N-of-1 Insights")
        add_image(ax, "docs/screenshots/overview.png", 0.1, 0.1, 0.8, 0.65, "Live Dashboard running on Heroku")
        pdf.savefig(fig)
        plt.close()

        # SLIDE 4: The Math (Proof of Rigor)
        fig, ax = setup_slide("You Can't Fake Math")
        ax.text(0.1, 0.75, "We validate relationships mathematically before the AI sees them.", color=THEME['text'], fontsize=16)
        
        # Render Math
        math_text = (
            r"$\bf{Pearson\ Correlation:}\ r = \frac{\sum(x-\bar{x})(y-\bar{y})}{\sqrt{\sum(x-\bar{x})^2 \sum(y-\bar{y})^2}}$"
            "\n\n"
            r"$\bf{Markov\ Transition:}\ P_{ij} = P(S_{t+1}=j \mid S_t=i)$"
            "\n\n"
            r"$\bf{AR(1)\ Model:}\ X_t = c + \phi X_{t-1} + \epsilon_t$"
        )
        ax.text(0.1, 0.35, math_text, color=THEME['accent'], fontsize=18, fontname='DejaVu Sans Mono')
        
        add_image(ax, "docs/screenshots/correlations.png", 0.55, 0.2, 0.4, 0.55, "Computed Correlation Matrix")
        pdf.savefig(fig)
        plt.close()

        # SLIDE 5: Deep Dive (The new screenshot)
        fig, ax = setup_slide("Deep Dive Analysis")
        
        if os.path.exists("docs/screenshots/deep_dive.png"):
            add_image(ax, "docs/screenshots/deep_dive.png", 0.1, 0.1, 0.8, 0.65, "Distribution & Time-Series Analysis")
        else:
            # Fallback if user hasn't added it yet
            ax.text(0.5, 0.5, "Deep Dive Analysis Placeholder", color=THEME['dim'], ha='center', fontsize=20)
            ax.text(0.5, 0.45, "(Add docs/screenshots/deep_dive.png to see this slide)", color='red', ha='center', fontsize=12)

        pdf.savefig(fig)
        plt.close()

        # SLIDE 6: The Agent Team
        fig, ax = setup_slide("The Agent Team")
        points = [
            "5 Specialists (not 1 generic prompt).",
            "Read-Only Access via SQL tools.",
            "Long-term memory tracks previous advice.",
            "Built with CrewAI + Gemini 2.5 Flash."
        ]
        add_bullet_points(ax, points)
        add_image(ax, "docs/screenshots/agent_chat.png", 0.1, 0.05, 0.8, 0.45, "Multi-Agent Conversation")
        pdf.savefig(fig)
        plt.close()
        
        print(f"Successfully generated {output_pdf}")

if __name__ == "__main__":
    create_slides()
