"""Stile comune dei grafici del progetto."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib as mpl

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from microgrid.optimize import Plan

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


def plot_plan(plan: Plan, config: dict[str, Any], title: str = "") -> Figure:
    """Disegna il piano di un giorno: potenze previste, batteria, rete, stato di carica e prezzi.

    Ogni pannello ha una sola scala verticale. L'asse orizzontale è l'ora
    locale del giorno.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    schedule = plan.schedule
    hours = len(schedule)
    start = np.arange(hours)
    middle = start + 0.5
    capacity = config["battery"]["capacity_kwh"]

    figure, axes = plt.subplots(
        5, 1, figsize=(9, 11), sharex=True, gridspec_kw={"height_ratios": [1.4, 1, 1, 1, 1], "hspace": 0.38}
    )
    forecast, battery, grid, soc, price = axes

    forecast.step(np.arange(hours + 1), np.append(schedule["load_kw"], schedule["load_kw"].iloc[-1]), where="post", color=COLORS["load"], label="Carico")
    forecast.step(np.arange(hours + 1), np.append(schedule["pv_kw"], schedule["pv_kw"].iloc[-1]), where="post", color=COLORS["pv"], label="Fotovoltaico")
    forecast.set_title("Carico e fotovoltaico previsti [kW]")
    forecast.legend(loc="upper left", ncols=2)

    battery.bar(middle, schedule["discharge_kw"] - schedule["charge_kw"], width=0.8, color=COLORS["alt"])
    battery.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
    battery.set_title("Batteria [kW]: scarica (+), carica (−)")

    grid.bar(middle, schedule["grid_import_kw"] - schedule["grid_export_kw"], width=0.8, color=COLORS["muted"])
    grid.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
    curtailed = schedule["curtailment_kw"].sum()
    grid.set_title(
        "Rete [kW]: acquisto (+), vendita (−)"
        + (f" — fotovoltaico tagliato: {curtailed:.0f} kWh" if curtailed > 0 else "")
    )

    soc.plot(np.arange(hours + 1), np.append(plan.soc_initial_kwh, schedule["soc_kwh"]), color=COLORS["alt"], marker="o", markersize=3)
    for fraction in (config["battery"]["soc_min_frac"], config["battery"]["soc_max_frac"]):
        soc.axhline(fraction * capacity, color=TEXT_SECONDARY, linewidth=0.8, linestyle="--")
    soc.set_ylim(0, capacity)
    soc.set_title("Energia in batteria [kWh], con i limiti minimo e massimo")

    price.step(np.arange(hours + 1), np.append(schedule["buy_price_eur_kwh"], schedule["buy_price_eur_kwh"].iloc[-1]), where="post", color=COLORS["price"], label="Acquisto")
    price.step(np.arange(hours + 1), np.append(schedule["sell_price_eur_kwh"], schedule["sell_price_eur_kwh"].iloc[-1]), where="post", color=COLORS["muted"], label="Vendita")
    price.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
    price.set_title("Prezzi [EUR/kWh]")
    price.legend(loc="center left", ncols=2)
    price.set_xlabel("Ora locale")
    price.set_xlim(0, hours)
    price.set_xticks(np.arange(0, hours + 1, 3))

    if title:
        figure.suptitle(title, x=0.125, y=0.93, ha="left", fontsize=12, fontweight="bold")
    return figure
