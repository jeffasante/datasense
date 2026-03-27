from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.markdown import Markdown
from rich import box
from typing import Dict, Any, List

class DashboardTUI:
    """Renders a beautiful, interactive terminal dashboard for analysis results."""

    def __init__(self):
        self.console = Console()

    def render(self, result: Dict[str, Any]):
        """Renders the complete analysis dashboard."""
        
        # Header
        header = Panel(
            "[bold #0056D2]DataSense — Multimodal Architecture Advisor[/bold #0056D2]",
            style="bold white",
            box=box.DOUBLE
        )
        
        # Fingerprint Summary
        fp_table = Table(title="Dataset Fingerprint", box=box.ROUNDED)
        fp_table.add_column("Modality", style="cyan")
        fp_table.add_column("Samples", justify="right", style="green")
        fp_table.add_column("Key Signal", style="magenta")
        
        fingerprints = result.get("fingerprints", {})
        for mod, fp in fingerprints.items():
            if mod == "mixed": continue
            count = fp.get("sample_count", 0)
            signals = fp.get("signals", [])
            primary_signal = signals[0] if signals else "Stable Distribution"
            fp_table.add_row(mod.capitalize(), str(count), primary_signal)

        # Leaderboard
        lb_table = Table(title="Architecture Leaderboard", box=box.HEAVY)
        lb_table.add_column("#", style="dim")
        lb_table.add_column("Model", style="bold white")
        lb_table.add_column("Score", justify="right")
        lb_table.add_column("Strategic Justification")
        
        leaderboard = result.get("leaderboard", [])
        for i, entry in enumerate(leaderboard[:10]):
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
            score = int(float(entry["score"]) * 100)
            style = "bold green" if score > 90 else "white"
            lb_table.add_row(f"{medal}{i+1}", entry["model"], f"[{style}]{score}%[/]", entry["justification"])

        # Executive Summary - Styled as Markdown for clean terminal output
        summary_raw = result.get("summary", "No summary available.")
        summary_content = Markdown(summary_raw)
        
        summary_panel = Panel(
            summary_content,
            title="Executive Summary & Reasoning",
            box=box.ROUNDED,
            border_style="cyan",
            padding=(1, 2)
        )

        # Build final view
        self.console.print(header)
        self.console.print(fp_table)
        self.console.print(lb_table)
        self.console.print(summary_panel)
