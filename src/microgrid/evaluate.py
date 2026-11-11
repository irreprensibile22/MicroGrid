"""Metriche e confronti fra strategie.

Uso:

    python -m microgrid.evaluate

Simula tutte le strategie sullo stesso periodo e scrive in `results/`:

- `comparison.csv`   costo ed emissioni totali di ogni strategia;
- `daily_costs.csv`  costo giornaliero di ogni strategia ed errori di previsione;
- `pareto.csv`       costo ed emissioni al variare del peso `λ`;
- `sensitivity.csv`  risparmio al variare di batteria, fotovoltaico e oneri.
"""

from __future__ import annotations

import copy
from typing import Any

import pandas as pd

from microgrid.config import load_config, resolve_path
from microgrid.dataset import local_days
from microgrid.forecast import load_forecasts
from microgrid.optimize import battery_limits, exchange_prices
from microgrid.simulate import simulate, simulate_rule, simulation_inputs

COMPARISON_FILE = "comparison.csv"
DAILY_FILE = "daily_costs.csv"
PARETO_FILE = "pareto.csv"
SENSITIVITY_FILE = "sensitivity.csv"

# Nome breve -> descrizione. L'ordine è quello delle tabelle e dei grafici.
STRATEGIES = {
    "no_battery": "Nessuna batteria",
    "rule": "Regola ingenua",
    "milp": "Previsione + MILP",
    "milp_corrected": "Previsione + MILP + correzione",
    "perfect": "Previsione perfetta + MILP",
}
# Variante: come `milp_corrected`, ma con la previsione ingenua del carico.
VARIANTS = {"milp_corrected_naive_load": "Previsione ingenua del carico + MILP + correzione"}
# Strategie che dipendono dal peso delle emissioni.
OPTIMIZED = ("milp", "milp_corrected", "perfect")
# Parametro della sensitività -> (sezione, chiave) nella configurazione.
SENSITIVITY_KEYS = {
    "battery_capacity_kwh": ("battery", "capacity_kwh"),
    "pv_peak_kw": ("pv", "peak_kw"),
    "import_surcharge_eur_kwh": ("grid", "import_surcharge_eur_kwh"),
}


def run_strategy(
    name: str, inputs: pd.DataFrame, config: dict[str, Any], carbon_weight: float | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simula una strategia; restituisce dettaglio orario e serie giornaliera."""
    if name == "no_battery":
        return simulate_rule(inputs, config, use_battery=False)
    if name == "rule":
        return simulate_rule(inputs, config, use_battery=True)
    if name == "milp":
        return simulate(inputs, config, carbon_weight=carbon_weight)
    if name == "milp_corrected":
        return simulate(inputs, config, carbon_weight=carbon_weight, correct=True)
    if name == "milp_corrected_naive_load":
        return simulate(inputs, config, load_column="load_naive", carbon_weight=carbon_weight, correct=True)
    if name == "perfect":
        return simulate(inputs, config, load_column="load_kw", pv_column="pv_kw", carbon_weight=carbon_weight)
    raise ValueError(f"Strategia sconosciuta: {name!r}")


def totals(daily: pd.DataFrame, inputs: pd.DataFrame, config: dict[str, Any]) -> dict[str, float]:
    """Totali di una simulazione sull'intero periodo.

    `residual_value_eur` è il valore dell'energia in più (o in meno) rimasta
    in batteria rispetto all'inizio, al prezzo medio di acquisto: serve a
    confrontare strategie che finiscono con la batteria a livelli diversi.
    """
    battery, limits = config["battery"], battery_limits(config)
    usable = limits["soc_max_kwh"] - limits["soc_min_kwh"]
    buy_price, _ = exchange_prices(inputs["price_eur_kwh"], config)
    stored = daily["soc_end_kwh"].iloc[-1] - daily["soc_start_kwh"].iloc[0]
    return {
        "cost_eur": daily["cost_eur"].sum(),
        "emissions_kg": daily["emissions_kg"].sum(),
        "import_kwh": daily["import_kwh"].sum(),
        "export_kwh": daily["export_kwh"].sum(),
        "curtailed_kwh": daily["curtailed_kwh"].sum(),
        "battery_cycles": daily["battery_discharge_kwh"].sum() / usable if usable > 0 else 0.0,
        "soc_end_kwh": daily["soc_end_kwh"].iloc[-1],
        "residual_value_eur": stored * battery["eta_discharge"] * buy_price.mean(),
    }


def add_savings(table: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge i risparmi rispetto a `no_battery` e la quota del risparmio teorico.

    `saving_captured` è il risparmio di costo diviso per quello della
    previsione perfetta: 1 significa che la strategia ottiene tutto ciò che
    si potrebbe ottenere conoscendo il futuro.
    """
    table = table.copy()
    baseline, perfect = table.loc["no_battery"], table.loc["perfect"]
    table["cost_saving_eur"] = baseline["cost_eur"] - table["cost_eur"]
    table["cost_saving_pct"] = 100 * table["cost_saving_eur"] / abs(baseline["cost_eur"])
    table["emissions_saving_pct"] = 100 * (baseline["emissions_kg"] - table["emissions_kg"]) / baseline["emissions_kg"]
    table["saving_captured"] = table["cost_saving_eur"] / (baseline["cost_eur"] - perfect["cost_eur"])
    return table


def compare(
    inputs: pd.DataFrame, config: dict[str, Any], strategies: tuple[str, ...] = tuple(STRATEGIES)
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Confronta le strategie; restituisce la tabella riassuntiva e i costi giornalieri."""
    rows, daily_costs = {}, {}
    for name in strategies:
        _, daily = run_strategy(name, inputs, config)
        rows[name] = totals(daily, inputs, config)
        daily_costs[name] = daily["cost_eur"]
    table = add_savings(pd.DataFrame(rows).T.rename_axis("strategy"))
    return table, pd.DataFrame(daily_costs).rename_axis("day")


def forecast_errors(inputs: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Errore di previsione di ogni giorno: MAE orario [kW] ed errore sull'energia [kWh]."""
    step = float(config["time"]["step_hours"])
    days = local_days(inputs.index, config).to_numpy()
    errors = {}
    for name in ("load", "pv"):
        error = inputs[f"{name}_forecast"] - inputs[f"{name}_kw"]
        errors[f"{name}_mae_kw"] = error.abs().groupby(days).mean()
        errors[f"{name}_error_kwh"] = error.groupby(days).sum() * step
    frame = pd.DataFrame(errors)
    frame.index = [day.date().isoformat() for day in frame.index]
    return frame.rename_axis("day")


def pareto(inputs: pd.DataFrame, config: dict[str, Any], strategies: tuple[str, ...] = OPTIMIZED) -> pd.DataFrame:
    """Costo ed emissioni reali al variare del peso `λ` delle emissioni."""
    rows = []
    for weight in config["evaluate"]["pareto_carbon_weights"]:
        for name in strategies:
            _, daily = run_strategy(name, inputs, config, carbon_weight=weight)
            rows.append(
                {
                    "carbon_weight_eur_kg": weight,
                    "strategy": name,
                    "cost_eur": daily["cost_eur"].sum(),
                    "emissions_kg": daily["emissions_kg"].sum(),
                }
            )
    return pd.DataFrame(rows)


def with_parameter(
    inputs: pd.DataFrame, config: dict[str, Any], parameter: str, value: float
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Copia di dati e configurazione con un parametro dell'impianto cambiato.

    Cambiando la potenza del fotovoltaico, produzione reale e prevista sono
    riscalate in proporzione.
    """
    section, key = SENSITIVITY_KEYS[parameter]
    changed = copy.deepcopy(config)
    changed[section][key] = value
    if parameter == "pv_peak_kw":
        scale = value / config["pv"]["peak_kw"]
        inputs = inputs.assign(pv_kw=inputs["pv_kw"] * scale, pv_forecast=inputs["pv_forecast"] * scale)
    return inputs, changed


def sensitivity(inputs: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Risultati di ogni strategia cambiando un parametro alla volta."""
    frames = []
    for parameter, values in config["evaluate"]["sensitivity"].items():
        for value in values:
            table, _ = compare(*with_parameter(inputs, config, parameter, value))
            table.insert(0, "value", value)
            table.insert(0, "parameter", parameter)
            frames.append(table.reset_index())
    columns = [
        "parameter",
        "value",
        "strategy",
        "cost_eur",
        "emissions_kg",
        "cost_saving_eur",
        "cost_saving_pct",
        "saving_captured",
    ]
    return pd.concat(frames, ignore_index=True)[columns]


def evaluation_inputs(config: dict[str, Any]) -> pd.DataFrame:
    """Dati della simulazione più la previsione ingenua del carico, per la variante."""
    inputs = simulation_inputs(config, config["simulate"]["split"])
    inputs["load_naive"] = load_forecasts(config).loc[inputs.index, "load_naive"]
    return inputs


def main() -> None:
    """Confronta le strategie e salva le tabelle (comando `python -m microgrid.evaluate`)."""
    config = load_config()
    inputs = evaluation_inputs(config)
    results_dir = resolve_path(config["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)

    table, daily_costs = compare(inputs, config, strategies=(*STRATEGIES, *VARIANTS))
    table.insert(0, "description", [{**STRATEGIES, **VARIANTS}[name] for name in table.index])
    table.to_csv(results_dir / COMPARISON_FILE, float_format="%.4f")
    daily_costs.join(forecast_errors(inputs, config)).to_csv(results_dir / DAILY_FILE, float_format="%.4f")
    print(table.drop(columns="description").round(2).to_string(), flush=True)

    front = pareto(inputs, config)
    front.to_csv(results_dir / PARETO_FILE, index=False, float_format="%.4f")
    print(front.round(2).to_string(index=False), flush=True)

    sweep = sensitivity(inputs, config)
    sweep.to_csv(results_dir / SENSITIVITY_FILE, index=False, float_format="%.4f")
    print(sweep.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
