import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
import matplotlib.image as mpimg
from PIL import Image
import os

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

def add_bullet_points(ax, points, x=0.1, y=0.75, spacing=0.08):
    for pt in points:
        ax.text(x, y, "• " + pt, color=THEME['text'], fontsize=18, va='top', fontname='Arial')
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

    img = Image.open(path)
    ax.imshow(img, extent=(x, x+width, y, y+height), aspect='auto', zorder=1)
    
    # Border
    rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor=THEME['accent'], facecolor='none', zorder=2)
    ax.add_patch(rect)
    
    if caption:
        ax.text(x + width/2, y - 0.05, caption, color=THEME['dim'], ha='center', fontsize=12, style='italic')

def create_slides():
    output_pdf = "docs/garmin_engineering_deep_dive.pdf"
    
    with PdfPages(output_pdf) as pdf:
        
        # SLIDE 1: Title
        fig, ax = setup_slide("Garmin Health Intelligence")
        ax.text(0.5, 0.6, "1,640 Lines of Math.\n5 AI Agents.\n0 Hallucinations.", 
                color=THEME['text'], fontsize=40, ha='center', fontweight='bold')
        ax.text(0.5, 0.4, "An Engineering approach to Personal Health Analytics", 
                color=THEME['secondary'], fontsize=18, ha='center')
        
        tech_stack = "Python • PostgreSQL • CrewAI • Streamlit • Scipy • GitHub Actions"
        ax.text(0.5, 0.2, tech_stack, color=THEME['dim'], fontsize=14, ha='center', fontname='Consolas')
        pdf.savefig(fig)
        plt.close()

        # SLIDE 2: The Core Problem
        fig, ax = setup_slide("The Problem with 'Chat with your Data'")
        points = [
            "LLMs are probabilistic token predictors, not calculators.",
            "Feeding raw CSVs to GPT-4 leads to hallucinations.",
            "They invent correlations that sound plausible but are false.",
            "We need a deterministic foundation before semantics."
        ]
        add_bullet_points(ax, points)
        
        # Visual diagram (simulated)
        diagram = """
        ❌  Naive Approach:
        [ Raw CSV ] -> [ LLM Prompt ] -> "Your coffee intake predicts rain." (HALLUCINATION)
        
        ✅  Engineering Approach:
        [ Raw CSV ] -> [ Scipy/Pandas Engine ] -> [ Validated Stats ] -> [ LLM ] -> "True Insight"
        """
        add_code_block(ax, diagram, x=0.15, y=0.45, fontsize=11)
        pdf.savefig(fig)
        plt.close()

        # SLIDE 3: The Architecture
        fig, ax = setup_slide("System Architecture")
        pipeline = """
        1. DATA INGESTION (GitHub Actions)
           Garmin API -> PostgreSQL (Upsert)
        
        2. DETERMINISTIC MATH LAYER (Python/Numpy)
           • Pearson Correlations (Linear)
           • AR(1) Models (Persistence)
           • Markov Chains (State Transitions)
           • Anomaly Detection (Percentiles)
        
        3. AGENTIC LAYER (CrewAI)
           • 5 Specialized Agents (Read-Only)
           • Synthsizer with Long-Term Memory
        """
        add_code_block(ax, pipeline, x=0.1, y=0.75, fontsize=12)
        add_image(ax, "docs/screenshots/overview.png", 0.55, 0.2, 0.4, 0.6, "Live Streamlit Dashboard")
        pdf.savefig(fig)
        plt.close()

        # SLIDE 4: The Math (Proof of Rigor)
        fig, ax = setup_slide("Determinism Before Semantics")
        ax.text(0.1, 0.75, "We validate relationships mathematically before the AI sees them.", color=THEME['text'], fontsize=16)
        
        # Render Math
        math_text = (
            r"$\bf{Pearson\ Correlation:}\ r = \frac{\sum(x-\bar{x})(y-\bar{y})}{\sqrt{\sum(x-\bar{x})^2 \sum(y-\bar{y})^2}}$"
            "\n\n\n"
            r"$\bf{Markov\ Transition:}\ P_{ij} = P(S_{t+1}=j \mid S_t=i)$"
            "\n\n\n"
            r"$\bf{AR(1)\ Model:}\ X_t = c + \phi X_{t-1} + \epsilon_t$"
        )
        ax.text(0.1, 0.4, math_text, color=THEME['accent'], fontsize=18, fontname='DejaVu Sans Mono')
        
        add_image(ax, "docs/screenshots/correlations.png", 0.55, 0.2, 0.4, 0.5, "Computed Correlation Matrix")
        pdf.savefig(fig)
        plt.close()

        # SLIDE 5: Agent Chat
        fig, ax = setup_slide("The Agent Team")
        points = [
            "5 Specialists (Stats, Sleep, Recovery, Patterns, Lead).",
            "Read-Only SQL Access (Sandboxed).",
            "Pre-computed context injected into prompt.",
            "Long-term memory tracks recommendation success."
        ]
        add_bullet_points(ax, points)
        add_image(ax, "docs/screenshots/agent_chat.png", 0.1, 0.05, 0.8, 0.45, "Multi-Agent Conversation")
        pdf.savefig(fig)
        plt.close()

        # SLIDE 6: Deep Dive (Optional)
        fig, ax = setup_slide("Deep Dive: Signal vs Noise")
        if os.path.exists("docs/screenshots/deep_dive.png"):
            add_image(ax, "docs/screenshots/deep_dive.png", 0.1, 0.1, 0.8, 0.65, "Distribution and Time-Series Analysis")
        else:
            ax.text(0.5, 0.5, "Placeholder for Deep Dive Analysis", color=THEME['dim'], ha='center')
            points = [
                "Full distribution analysis (histogram + KDE).",
                "Time-series decomposition.",
                "Moving averages (7d/14d) to spot true trends."
            ]
            add_bullet_points(ax, points, x=0.1, y=0.7)
            
        pdf.savefig(fig)
        plt.close()
        
        print(f"Successfully generated {output_pdf}")

if __name__ == "__main__":
    create_slides()
