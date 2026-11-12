"""Stile comune dei grafici del progetto."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib as mpl

if TYPE_CHECKING:
    import pandas as pd
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

    forecast.step(
        np.arange(hours + 1),
        np.append(schedule["load_kw"], schedule["load_kw"].iloc[-1]),
        where="post",
        color=COLORS["load"],
        label="Carico",
    )
    forecast.step(
        np.arange(hours + 1),
        np.append(schedule["pv_kw"], schedule["pv_kw"].iloc[-1]),
        where="post",
        color=COLORS["pv"],
        label="Fotovoltaico",
    )
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

    soc.plot(
        np.arange(hours + 1),
        np.append(plan.soc_initial_kwh, schedule["soc_kwh"]),
        color=COLORS["alt"],
        marker="o",
        markersize=3,
    )
    for fraction in (config["battery"]["soc_min_frac"], config["battery"]["soc_max_frac"]):
        soc.axhline(fraction * capacity, color=TEXT_SECONDARY, linewidth=0.8, linestyle="--")
    soc.set_ylim(0, capacity)
    soc.set_title("Energia in batteria [kWh], con i limiti minimo e massimo")

    price.step(
        np.arange(hours + 1),
        np.append(schedule["buy_price_eur_kwh"], schedule["buy_price_eur_kwh"].iloc[-1]),
        where="post",
        color=COLORS["price"],
        label="Acquisto",
    )
    price.step(
        np.arange(hours + 1),
        np.append(schedule["sell_price_eur_kwh"], schedule["sell_price_eur_kwh"].iloc[-1]),
        where="post",
        color=COLORS["muted"],
        label="Vendita",
    )
    price.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
    price.set_title("Prezzi [EUR/kWh]")
    price.legend(loc="center left", ncols=2)
    price.set_xlabel("Ora locale")
    price.set_xlim(0, hours)
    price.set_xticks(np.arange(0, hours + 1, 3))

    if title:
        figure.suptitle(title, x=0.125, y=0.93, ha="left", fontsize=12, fontweight="bold")
    return figure


# Colore e tratto di ogni strategia, uguali in tutti i grafici dei risultati.
# I due riferimenti (nessuna batteria e previsione perfetta) sono neutri; le
# tre strategie realizzabili hanno un colore.
STRATEGY_STYLES = {
    "no_battery": {"color": COLORS["muted"], "linestyle": "-"},
    "rule": {"color": COLORS["pv"], "linestyle": "-"},
    "milp": {"color": COLORS["load"], "linestyle": "-"},
    "milp_corrected": {"color": COLORS["alt"], "linestyle": "-"},
    "perfect": {"color": TEXT_SECONDARY, "linestyle": "--"},
}


def _thousands(value: float) -> str:
    """Numero intero con lo spazio come separatore delle migliaia."""
    return f"{value:,.0f}".replace(",", " ")


def plot_comparison(comparison: pd.DataFrame, labels: dict[str, str]) -> Figure:
    """Costo totale di ogni strategia di `labels`, con il risparmio rispetto a nessuna batteria."""
    import matplotlib.pyplot as plt
    import numpy as np

    table = comparison.loc[list(labels)]
    figure, ax = plt.subplots(figsize=(9, 3.4))
    positions = np.arange(len(table))[::-1]
    for position, (name, row) in zip(positions, table.iterrows(), strict=True):
        ax.barh(position, row["cost_eur"], height=0.62, color=STRATEGY_STYLES[name]["color"])
        saving = "" if name == "no_battery" else f"   −{row['cost_saving_pct']:.1f}%".replace(".", ",")
        ax.text(row["cost_eur"] + 60, position, f"{_thousands(row['cost_eur'])} €{saving}", va="center", fontsize=9.5)
    ax.set_yticks(positions, [labels[name] for name in table.index])
    ax.set_xlim(0, table["cost_eur"].max() * 1.22)
    ax.set_xlabel("costo netto sul test set (€)")
    ax.set_title("Costo totale per strategia, e risparmio rispetto a nessuna batteria")
    ax.grid(axis="y", visible=False)
    return figure


def plot_pareto(front: pd.DataFrame, labels: dict[str, str]) -> Figure:
    """Costo ed emissioni al variare di λ, un pannello per strategia.

    I pannelli hanno la stessa ampiezza degli assi (400 kgCO2 e 120 EUR),
    così le curve sono confrontabili anche se lontane fra loro.
    """
    import matplotlib.pyplot as plt

    names = [name for name in labels if name in set(front["strategy"])]
    figure, axes = plt.subplots(1, len(names), figsize=(11, 3.9))
    for ax, name in zip(axes, names, strict=True):
        curve = front[front["strategy"] == name]
        ax.plot(curve["emissions_kg"], curve["cost_eur"], marker="o", markersize=5, **STRATEGY_STYLES[name])
        for row in curve.itertuples():
            ax.annotate(
                f"{row.carbon_weight_eur_kg:g}",
                (row.emissions_kg, row.cost_eur),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8.5,
                color=TEXT_SECONDARY,
            )
        x_mid = (curve["emissions_kg"].min() + curve["emissions_kg"].max()) / 2
        y_mid = (curve["cost_eur"].min() + curve["cost_eur"].max()) / 2
        ax.set_xlim(x_mid - 200, x_mid + 200)
        ax.set_ylim(y_mid - 60, y_mid + 60)
        ax.set_xlabel("emissioni (kgCO₂)")
        ax.set_title(labels[name], fontsize=10)
    axes[0].set_ylabel("costo (€)")
    figure.suptitle(
        "Costo ed emissioni al variare di λ (numeri accanto ai punti, in €/kgCO₂)",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    figure.tight_layout()
    return figure


def plot_sensitivity(sweep: pd.DataFrame, labels: dict[str, str]) -> Figure:
    """Risparmio rispetto a nessuna batteria cambiando un parametro alla volta."""
    import matplotlib.pyplot as plt

    panels = [
        ("battery_capacity_kwh", "Capacità della batteria (kWh)"),
        ("pv_peak_kw", "Potenza del fotovoltaico (kWp)"),
        ("import_surcharge_eur_kwh", "Oneri sull'acquisto (€/kWh)"),
    ]
    names = [name for name in labels if name != "no_battery"]
    figure, axes = plt.subplots(1, len(panels), figsize=(11.5, 4), sharey=True)
    for ax, (parameter, title) in zip(axes, panels, strict=True):
        part = sweep[sweep["parameter"] == parameter]
        for name in names:
            curve = part[part["strategy"] == name]
            ax.plot(
                curve["value"],
                curve["cost_saving_eur"],
                marker="o",
                markersize=4,
                label=labels[name],
                **STRATEGY_STYLES[name],
            )
        ax.set_xticks(sorted(part["value"].unique()))
        ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
        ax.set_xlabel(title)
    axes[0].set_ylabel("risparmio rispetto a nessuna batteria (€)")
    axes[0].legend(loc="upper left", fontsize=8)
    figure.suptitle(
        "Risparmio al variare di un parametro alla volta", x=0.01, ha="left", fontsize=11, fontweight="bold"
    )
    figure.tight_layout()
    return figure
