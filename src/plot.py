import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

# Set matplotlib style and font
plt.rcParams["figure.dpi"] = 300  # High-resolution output
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.weight"] = "bold"

# Colors from your style
CYAN = "#00C4E6"
ORANGE = "#F28C38"
LIGHT_ORANGE = "#F5B270"
GREEN = "#2CA02C"
DARK_BG = "#1A2526"
WHITE = "#FFFFFF"

def simple_duration_histogram(merged_df, bins=30):
    """
    Simple, clean histogram for quick analysis
    """
    duration_data = merged_df['duration_months'].dropna()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    
    # Create histogram
    n, bins_edges, patches = ax.hist(duration_data, bins=bins, alpha=0.7, color=CYAN, edgecolor=WHITE)
    
    # Add statistics
    ax.axvline(duration_data.mean(), color=ORANGE, linestyle='--', linewidth=2, 
               label=f'Média: {duration_data.mean():.1f} meses')
    ax.axvline(duration_data.median(), color=GREEN, linestyle='--', linewidth=2, 
               label=f'Mediana: {duration_data.median():.1f} meses')
    
    ax.set_title('Distribuição de Duração do Ciclo de Vida', color=WHITE, fontsize=14, fontweight='bold')
    ax.set_xlabel('Duração (meses)', color=WHITE, fontsize=12)
    ax.set_ylabel('Frequência', color=WHITE, fontsize=12)
    
    # Style
    ax.grid(True, alpha=0.3, color=WHITE)
    ax.tick_params(colors=WHITE)
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Legend
    legend = ax.legend(fontsize=10, loc='upper right')
    legend.get_frame().set_facecolor(DARK_BG)
    legend.get_frame().set_edgecolor(WHITE)
    for text in legend.get_texts():
        text.set_color(WHITE)
    
    # Add statistics box
    stats_text = f"N = {len(duration_data)}\nDesvio Padrão = {duration_data.std():.1f}\nMín = {duration_data.min():.1f}\nMáx = {duration_data.max():.1f}"
    props = dict(boxstyle='round,pad=0.5', facecolor=DARK_BG, edgecolor=CYAN, alpha=0.8)
    ax.text(0.75, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', color=CYAN, bbox=props)
    
    plt.tight_layout()
    plt.show()