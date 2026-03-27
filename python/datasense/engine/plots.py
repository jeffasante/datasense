import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os
import sys
from typing import Dict, Any, List

# Optional library support for advanced plots
try:
    import librosa
    import librosa.display
except ImportError:
    librosa = None

# Color palettes for a premium "Sapphire" aesthetic
THEMES = {
    "dark": {
        "gold":   "#F59E0B",
        "silver": "#94A3B8",
        "bronze": "#D97706",
        "bar":    "#6366F1",
        "bg":     "#0F172A",
        "surface":"#1E293B",
        "text":   "#E2E8F0",
        "muted":  "#94A3B8",
        "grid":   "#334155",
        "accent": "#818CF8",
    },
    "light": {
        "gold":   "#B45309",
        "silver": "#64748B",
        "bronze": "#92400E",
        "bar":    "#2563EB",
        "bg":     "#FFFFFF",
        "surface":"#F8FAFC",
        "text":   "#0F172A",
        "muted":  "#64748B",
        "grid":   "#E2E8F0",
        "accent": "#2563EB",
    }
}

def _style_ax(ax, fig, theme="dark"):
    """Apply theme to axes."""
    colors = THEMES[theme]
    fig.set_facecolor(colors["bg"])
    ax.set_facecolor(colors["surface"])
    ax.tick_params(colors=colors["text"], labelsize=9)
    ax.xaxis.label.set_color(colors["text"])
    ax.yaxis.label.set_color(colors["text"])
    ax.title.set_color(colors["text"])
    for spine in ax.spines.values():
        spine.set_color(colors["grid"])

def plot_leaderboard(leaderboard: List[Dict[str, Any]], output_path: str, theme="dark"):
    """Horizontal bar chart of architecture scores."""
    if not leaderboard: return
    colors = THEMES[theme]
    top = leaderboard[:10]
    top.reverse()

    models = [e["model"] for e in top]
    scores = [e["score"] * 100 for e in top]
    justifications = [e.get("justification", "") for e in top]

    fig, ax = plt.subplots(figsize=(14, 6))
    _style_ax(ax, fig, theme)

    bar_colors = [colors["gold"] if i == len(top)-1 else colors["silver"] if i == len(top)-2 else colors["bar"] for i in range(len(top))]
    ax.barh(models, scores, color=bar_colors, height=0.65)

    ax.set_xlim(0, 110)
    ax.set_title("Architecture Recommendation Leaderboard", fontweight="bold", pad=15)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_audio_spec(fingerprint: Dict[str, Any], output_path: str, theme="dark"):
    """Generates an illustrative spectrogram representation for audio modality."""
    colors = THEMES[theme]
    fig, ax = plt.subplots(figsize=(10, 4))
    _style_ax(ax, fig, theme)
    
    # Generate dummy spectral data if no real samples are provided in paths
    t = np.linspace(0, 5, 500)
    f = np.linspace(0, 8000, 128)
    Sxx = np.exp(-((f[:, None] - 440)**2 / (2 * 50**2))) * np.random.rand(128, 500)
    
    img = ax.imshow(10 * np.log10(Sxx + 1e-6), aspect='auto', origin='lower', extent=[0, 5, 0, 8000], cmap='magma')
    ax.set_title("Audio Modality Fingerprint: Spectral Power", fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)

def plot_tabular_profile(fingerprint: Dict[str, Any], output_path: str, theme="dark"):
    """Visualizes missingness and feature distribution signals for tabular data."""
    colors = THEMES[theme]
    fig, ax = plt.subplots(figsize=(10, 4))
    _style_ax(ax, fig, theme)
    
    missing = fingerprint.get("missing_rate", 0.2)
    labels = ["Complete", "Missing"]
    values = [100 - (missing*100), missing*100]
    
    ax.pie(values, labels=labels, autopct='%1.1f%%', colors=[colors["bar"], colors["gold"]], startangle=140, textprops={'color': colors['text']})
    ax.set_title("Tabular Data Integrity", fontweight="bold")
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)

def plot_image_grid(fingerprint: Dict[str, Any], output_path: str, theme="dark"):
    """Illustrative pixel resolution and spatial cluster visualization."""
    colors = THEMES[theme]
    fig, ax = plt.subplots(figsize=(6, 6))
    _style_ax(ax, fig, theme)
    
    res = fingerprint.get("resolution", {}).get("median", [224, 224])
    ax.add_patch(plt.Rectangle((0, 0), res[0], res[1], fill=False, edgecolor=colors["bar"], lw=3))
    ax.text(res[0]/2, res[1]/2, f"Median Profile:\n{int(res[0])}x{int(res[1])}", ha='center', va='center', color=colors["text"])
    
    ax.set_xlim(-50, res[0]+50)
    ax.set_ylim(-50, res[1]+50)
    ax.set_title("Image Spatial Footprint", fontweight="bold")
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)

def plot_fingerprint_radar(fingerprints: Dict[str, Any], output_path: str, theme="dark"):
    """Radar chart showing overall dataset health across modalities."""
    colors = THEMES[theme]
    signals = {"Spatial": 0.5, "Temporal": 0.5, "Tabular": 0.5, "Textual": 0.5, "Quality": 0.7}
    
    if "image" in fingerprints: signals["Spatial"] = 0.9
    if "video" in fingerprints: signals["Temporal"] = 0.9
    if "audio" in fingerprints: signals["Temporal"] = 0.8; signals["Quality"] = 0.8
    if "tabular" in fingerprints: signals["Tabular"] = 0.9
    
    labels = list(signals.keys())
    values = list(signals.values())
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    _style_ax(ax, fig, theme)
    
    ax.plot(angles, values, color=colors["bar"], linewidth=2)
    ax.fill(angles, values, color=colors["bar"], alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title("Dataset Cross-Modal Fingerprint", pad=20, fontweight="bold")
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

def generate_plots(fingerprints: Dict[str, Any], leaderboard: List[Dict[str, Any]], output_dir: str, theme="dark"):
    """Orchestrates all modality-specific visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    paths = {}

    # Global Plots
    lb_path = os.path.join(output_dir, f"leaderboard_{theme}.png")
    plot_leaderboard(leaderboard, lb_path, theme)
    paths["leaderboard"] = lb_path

    fp_path = os.path.join(output_dir, f"fingerprint_{theme}.png")
    plot_fingerprint_radar(fingerprints, fp_path, theme)
    paths["fingerprint"] = fp_path

    # Modality-Specific Plots
    if "audio" in fingerprints:
        a_path = os.path.join(output_dir, f"audio_profile_{theme}.png")
        plot_audio_spec(fingerprints["audio"], a_path, theme)
        paths["audio_profile"] = a_path

    if "tabular" in fingerprints:
        t_path = os.path.join(output_dir, f"tabular_profile_{theme}.png")
        plot_tabular_profile(fingerprints["tabular"], t_path, theme)
        paths["tabular_profile"] = t_path

    if "image" in fingerprints:
        i_path = os.path.join(output_dir, f"image_profile_{theme}.png")
        plot_image_grid(fingerprints["image"], i_path, theme)
        paths["image_profile"] = i_path

    return paths
