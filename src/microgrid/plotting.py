"""Stile comune dei grafici del progetto."""

from __future__ import annotations

import matplotlib as mpl

# Ogni grandezza ha sempre lo stesso colore, in tutti i grafici.
COLORS = {
    "load": "#2a78d6",
    "pv": "#eb6834",
    "alt": "#1baf7a",
    "price": "#6250d6",
    "carbon": "#008300",
    "muted": "#8a897f",
}
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3de"


def apply_style() -> None:
    """Imposta lo stile di matplotlib: linee sottili, griglia e assi discreti."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "font.size": 10,
            "text.color": TEXT,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.edgecolor": GRID,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "legend.frameon": False,
        }
    )
