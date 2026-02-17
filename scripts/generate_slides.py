import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
import matplotlib.lines as lines
from PIL import Image
import os
import matplotlib.font_manager as fm

# ==========================================
# DESIGN SYSTEM: "BENTO GRID 2026"
# ==========================================
SLIDE_W, SLIDE_H = 16, 9
DPI = 300

# Color Palette - "Technical" & "Premium"
COLORS = {
    'bg_canvas':  '#F3F4F6',   # Cool light grey background
    'card_bg':    '#FFFFFF',   # Pure white cards
    'text_main':  '#111827',   # Near Black
    'text_sec':   '#6B7280',   # Cool Grey
    'accent':     '#2563EB',   # Engineering Blue
    'border_subt': '#E5E7EB',  # Very subtle card borders
}

# Typography - System Stack
FONTS = {
    'sans': 'Segoe UI',
    'cond': 'Segoe UI', 
    'mono': 'Consolas',
}

# Type Scale 
TYPE = {
    'hero':       {'fontsize': 48, 'weight': 'bold', 'color': COLORS['text_main']},
    'h1':         {'fontsize': 32, 'weight': 'bold', 'color': COLORS['text_main']}, # Slide titles
    'h2':         {'fontsize': 24, 'weight': '600',  'color': COLORS['text_main']},
    'h3':         {'fontsize': 18, 'weight': '600',  'color': COLORS['text_main']}, # Sub-sections
    'label':      {'fontsize': 12, 'weight': '600',  'color': COLORS['accent']},
    'body':       {'fontsize': 14, 'weight': 'normal','color': COLORS['text_main']},
    'caption':    {'fontsize': 12, 'weight': 'normal','color': COLORS['text_sec']},
}

# ==========================================
# UI COMPONENTS
# ==========================================

def setup_slide():
    fig = plt.figure(figsize=(SLIDE_W, SLIDE_H), dpi=DPI, facecolor=COLORS['bg_canvas'])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(COLORS['bg_canvas'])
    ax.set_xlim(0, SLIDE_W)
    ax.set_ylim(0, SLIDE_H)
    ax.axis('off')
    return fig, ax

def draw_card(ax, x, y, w, h, title=None):
    """Draws a premium 'Bento' card with soft shadow"""
    # Shadows
    shadow1 = patches.FancyBboxPatch((x+0.1, y-0.1), w, h, boxstyle="round,pad=0", 
                                     fc='#000000', alpha=0.03, zorder=1)
    shadow2 = patches.FancyBboxPatch((x+0.05, y-0.05), w, h, boxstyle="round,pad=0", 
                                     fc='#000000', alpha=0.05, zorder=1)
    ax.add_patch(shadow1)
    ax.add_patch(shadow2)

    # Card
    card = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0", 
                                  fc=COLORS['card_bg'], ec=COLORS['border_subt'], lw=1, zorder=2)
    ax.add_patch(card)
    
    # Title
    if title:
        ax.text(x + 0.4, y + h - 0.5, title.upper(), 
                fontsize=11, fontweight='bold', color=COLORS['accent'], fontname=FONTS['sans'], 
                ha='left', va='center', zorder=5)

def add_header(ax, title):
    """Adds CENTERED slide header"""
    ax.text(SLIDE_W/2, 8.2, title.upper(), **TYPE['h1'], ha='center', va='bottom')
    # Optional centered decor line
    # ax.plot([SLIDE_W/2 - 1, SLIDE_W/2 + 1], [8.0, 8.0], color=COLORS['accent'], lw=3)

def place_image(ax, path, x, y, w, h, label=None):
    """Places image inside a card region"""
    if not os.path.exists(path):
        ax.text(x + w/2, y + h/2, f"MISSING: {os.path.basename(path)}", 
                ha='center', color='red', zorder=10)
        return

    try:
        img = Image.open(path)
        img_w, img_h = img.size
        
        # Container
        draw_card(ax, x, y, w, h, title=label)
        
        # Fit logic
        padding = 0.8 if label else 0.4 # Less padding if no label
        avail_w = w - padding
        avail_h = h - padding
        aspect = img_w / img_h
        
        if avail_w / aspect < avail_h:
            disp_w = avail_w
            disp_h = avail_w / aspect
        else:
            disp_h = avail_h
            disp_w = avail_h * aspect
            
        center_x = x + w/2
        center_y = y + h/2
        if label:
            center_y -= 0.1

        ax.imshow(img, extent=[center_x - disp_w/2, center_x + disp_w/2, 
                               center_y - disp_h/2, center_y + disp_h/2],
                  aspect='equal', zorder=6)
        
        # Border
        rect = patches.Rectangle((center_x - disp_w/2, center_y - disp_h/2), disp_w, disp_h, 
                                 linewidth=1, edgecolor=COLORS['border_subt'], facecolor='none', zorder=7)
        ax.add_patch(rect)

    except Exception as e:
        print(f"Error: {e}")

# ==========================================
# SLIDE LOGIC
# ==========================================

def create_slides():
    output = "docs/garmin_engineering_deep_dive.pdf"
    with PdfPages(output) as pdf:
        
        # --- SLIDE 1: HERO COVER ---
        fig, ax = setup_slide()
        
        # Center Card
        cw, ch = 12, 6
        cx, cy = (SLIDE_W-cw)/2, (SLIDE_H-ch)/2
        draw_card(ax, cx, cy, cw, ch)
        
        # Content - Title Only (No Subtitle)
        ax.text(SLIDE_W/2, SLIDE_H/2, "GARMIN HEALTH\nINTELLIGENCE", 
                **TYPE['hero'], ha='center', va='center')
        
        # Simple accent line below
        ax.plot([SLIDE_W/2 - 2, SLIDE_W/2 + 2], [SLIDE_H/2 - 1.5, SLIDE_H/2 - 1.5], 
                 color=COLORS['accent'], lw=4, zorder=10)
        
        pdf.savefig(fig)
        plt.close()
        
        # --- SLIDE 2: ARCHITECTURE ---
        fig, ax = setup_slide()
        add_header(ax, "System Architecture")
        
        # Left Card: Mission
        draw_card(ax, 0.8, 1.5, 6.8, 6.0, title="Mission Parameters")
        
        ly = 6.0
        step = 1.5
        items = [
            ("Probabilistic Engines", "LLMs generate plausible text,\nnot factual truth."),
            ("The Hallucination Gap", "Raw data + Prompts =\nUnreliable outputs."),
            ("Deterministic Fix", "Validated math constraints\nensure 100% accuracy.")
        ]
        for title, desc in items:
            ax.text(1.2, ly, title, **TYPE['h3'], ha='left')
            ax.text(1.2, ly - 0.5, desc, **TYPE['body'], ha='left', va='top')
            ly -= step

        # Right Card: Pipeline
        draw_card(ax, 8.4, 1.5, 6.8, 6.0, title="Pipeline Logic")
        
        # Nodes
        nx = 8.4 + 3.4 
        ny = 6.0
        nodes = ["Raw Sensor Data", "Deterministic Math", "Semantic Agents"]
        
        for i, node in enumerate(nodes):
            patches.FancyBboxPatch((nx-2, ny-0.4), 4, 0.8, boxstyle="round,pad=0.1", 
                                   fc='white', ec=COLORS['accent'], lw=2, zorder=5)
            # Fixed Color
            ax.text(nx, ny, node, **TYPE['h3'], ha='center', zorder=6)
            
            if i < len(nodes) - 1:
                ax.arrow(nx, ny-0.6, 0, -0.8, color=COLORS['text_sec'], head_width=0.15, length_includes_head=True, zorder=4)
            ny -= 1.5

        pdf.savefig(fig)
        plt.close()
        
        # --- SLIDE 3: DASHBOARD ---
        fig, ax = setup_slide()
        add_header(ax, "Operational Dashboard") # Centered
        # No label
        place_image(ax, "docs/screenshots/overview.png", 0.8, 1.0, 14.4, 6.5, label=None)
        pdf.savefig(fig)
        plt.close()

        # --- SLIDE 4: LOGIC ---
        fig, ax = setup_slide()
        add_header(ax, "Core Logic") # Centered
        
        # Left Stack
        stack_x = 0.8
        stack_w = 5.0
        stack_h = 1.8
        gap = 0.3
        sy = 1.5 + (stack_h + gap) * 2 
        
        logic_items = [
            ("Pearson Correlation", "Detects linear\nrelationships"),
            ("Markov Chains", "Predicts state\ntransitions"),
            ("AR(1) Models", "Separates trend\nfrom noise")
        ]
        
        for title, desc in logic_items:
            draw_card(ax, stack_x, sy, stack_w, stack_h, title=None)
            ax.text(stack_x + 0.3, sy + stack_h - 0.5, title, **TYPE['label'], ha='left')
            ax.text(stack_x + 0.3, sy + stack_h - 1.0, desc, **TYPE['body'], ha='left', va='top')
            sy -= (stack_h + gap)
            
        # Right: Heatmap (No Label)
        place_image(ax, "docs/screenshots/correlations.png", 6.2, 1.5, 9.0, 6.0, label=None)
        
        pdf.savefig(fig)
        plt.close()

        # --- SLIDE 5: AGENTS ---
        fig, ax = setup_slide()
        add_header(ax, "Multi-Agent System") # Centered
        place_image(ax, "docs/screenshots/agent_chat.png", 0.8, 1.0, 14.4, 6.5, label=None)
        pdf.savefig(fig)
        plt.close()
        
        # --- SLIDE 6: DEEP DIVE ---
        if os.path.exists("docs/screenshots/deep_dive.png"):
            fig, ax = setup_slide()
            add_header(ax, "Signal Analysis") # Centered
            place_image(ax, "docs/screenshots/deep_dive.png", 0.8, 1.0, 14.4, 6.5, label=None)
            pdf.savefig(fig)
            plt.close()

        print(f"Generated Final Clean Deck: {output}")

if __name__ == "__main__":
    create_slides()