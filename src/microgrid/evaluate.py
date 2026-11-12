"""Metrics and comparison between strategies.

Usage:

    python -m microgrid.evaluate

Simulates every strategy over the same period and writes to `results/`:

- `comparison.csv`   total cost and emissions of each strategy;
- `daily_costs.csv`  daily cost of each strategy, plus the forecast errors;
- `pareto.csv`       cost and emissions as the weight `λ` changes;
- `sensitivity.csv`  savings as battery, PV and surcharge change;
- `comparison.png`, `pareto.png`, `sensitivity.png`  the matching charts.

With `--figures-only` it just redraws the charts from the saved tables.
"""

from __future__ import annotations

import argparse
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
FIGURE_FILES = {"comparison": "comparison.png", "pareto": "pareto.png", "sensitivity": "sensitivity.png"}

# Short name -> description. The order is the one used in tables and charts.
STRATEGIES = {
    "no_battery": "No battery",
    "rule": "Naive rule",
    "milp": "Forecast + MILP",
    "milp_corrected": "Forecast + MILP + correction",
    "perfect": "Perfect forecast + MILP",
}
# Variant: like `milp_corrected`, but with the naive load forecast.
VARIANTS = {"milp_corrected_naive_load": "Naive load forecast + MILP + correction"}
# Strategies that depend on the emission weight.
OPTIMIZED = ("milp", "milp_corrected", "perfect")
# Sensitivity parameter -> (section, key) in the configuration.
SENSITIVITY_KEYS = {
    "battery_capacity_kwh": ("battery", "capacity_kwh"),
    "pv_peak_kw": ("pv", "peak_kw"),
    "import_surcharge_eur_kwh": ("grid", "import_surcharge_eur_kwh"),
}


def run_strategy(
    name: str, inputs: pd.DataFrame, config: dict[str, Any], carbon_weight: float | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate one strategy; return the hourly detail and the daily series."""
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
    raise ValueError(f"Unknown strategy: {name!r}")


def totals(daily: pd.DataFrame, inputs: pd.DataFrame, config: dict[str, Any]) -> dict[str, float]:
    """Totals of a simulation over the whole period.

    `residual_value_eur` is the value of the extra (or missing) energy left
    in the battery compared with the start, at the mean buy price. It makes
    strategies that end at different battery levels comparable.
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
    """Add the savings relative to `no_battery` and the share of the theoretical saving.

    `saving_captured` is the cost saving divided by that of the perfect
    forecast: 1 means the strategy gets everything that could be had by
    knowing the future.
    """
    table = table.copy()
    baseline, perfect = table.loc["no_battery"], table.loc["perfect"]
    table["cost_saving_eur"] = baseline["cost_eur"] - table["cost_eur"]
    # abs(): the baseline cost can be negative in sunny periods (net seller).
    table["cost_saving_pct"] = 100 * table["cost_saving_eur"] / abs(baseline["cost_eur"])
    table["emissions_saving_pct"] = 100 * (baseline["emissions_kg"] - table["emissions_kg"]) / baseline["emissions_kg"]
    table["saving_captured"] = table["cost_saving_eur"] / (baseline["cost_eur"] - perfect["cost_eur"])
    return table


def compare(
    inputs: pd.DataFrame, config: dict[str, Any], strategies: tuple[str, ...] = tuple(STRATEGIES)
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare the strategies; return the summary table and the daily costs."""
    rows, daily_costs = {}, {}
    for name in strategies:
        _, daily = run_strategy(name, inputs, config)
        rows[name] = totals(daily, inputs, config)
        daily_costs[name] = daily["cost_eur"]
    table = add_savings(pd.DataFrame(rows).T.rename_axis("strategy"))
    return table, pd.DataFrame(daily_costs).rename_axis("day")


def forecast_errors(inputs: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Forecast error of each day: hourly MAE [kW] and energy error [kWh]."""
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
    """Actual cost and emissions as the emission weight `λ` changes."""
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
    """Copy of data and configuration with one plant parameter changed.

    When the PV size changes, actual and forecast output are rescaled in
    proportion.
    """
    section, key = SENSITIVITY_KEYS[parameter]
    changed = copy.deepcopy(config)
    changed[section][key] = value
    if parameter == "pv_peak_kw":
        scale = value / config["pv"]["peak_kw"]
        inputs = inputs.assign(pv_kw=inputs["pv_kw"] * scale, pv_forecast=inputs["pv_forecast"] * scale)
    return inputs, changed


def sensitivity(inputs: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Results of every strategy, changing one parameter at a time."""
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
    """Simulation inputs plus the naive load forecast, needed by the variant."""
    inputs = simulation_inputs(config, config["simulate"]["split"])
    inputs["load_naive"] = load_forecasts(config).loc[inputs.index, "load_naive"]
    return inputs


def save_figures(config: dict[str, Any]) -> None:
    """Draw the result charts from the tables saved in `results/`."""
    # Local import: matplotlib is only needed here.
    from microgrid.plotting import apply_style, plot_comparison, plot_pareto, plot_sensitivity

    results_dir = resolve_path(config["paths"]["results"])
    apply_style()
    figures = {
        "comparison": plot_comparison(pd.read_csv(results_dir / COMPARISON_FILE, index_col="strategy"), STRATEGIES),
        "pareto": plot_pareto(pd.read_csv(results_dir / PARETO_FILE), STRATEGIES),
        "sensitivity": plot_sensitivity(pd.read_csv(results_dir / SENSITIVITY_FILE), STRATEGIES),
    }
    for name, figure in figures.items():
        figure.savefig(results_dir / FIGURE_FILES[name], dpi=150, bbox_inches="tight")


def main() -> None:
    """Compare the strategies and save tables and charts (`python -m microgrid.evaluate`)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--figures-only", action="store_true", help="redraw the charts from the saved tables")
    args = parser.parse_args()
    config = load_config()
    if args.figures_only:
        save_figures(config)
        return
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
    save_figures(config)


if __name__ == "__main__":
    main()
