"""MILP model for the hourly battery and grid schedule.

Usage:

    python -m microgrid.optimize    # solve the example day and plot its plan

The formulation is described in `docs/formulation.md`. The numbers in
brackets in the comments below are the constraint numbers used there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.contrib.solver.common.factory import SolverFactory
from pyomo.contrib.solver.common.results import SolutionStatus

from microgrid.config import load_config, resolve_path

EXAMPLE_FILE = "optimize_example.png"
# Solver values smaller than this are treated as zero.
ZERO_TOLERANCE = 1e-7
# Schedule column -> name of the variable in the model.
SCHEDULE_VARIABLES = {
    "charge_kw": "p_ch",
    "discharge_kw": "p_dis",
    "grid_import_kw": "g_imp",
    "grid_export_kw": "g_exp",
    "curtailment_kw": "curt",
    "soc_kwh": "soc",
}


class OptimizationError(RuntimeError):
    """The solver did not find an optimal solution."""


@dataclass(frozen=True)
class Plan:
    """Hourly plan for one day.

    `schedule` has one row per hour, with the same index as the input series.
    Powers [kW] are means over the hour; `soc_kwh` is the energy in the
    battery at the end of the hour. `cost_eur` and `emissions_kg` are what the
    plan expects, not what actually happens.
    """

    schedule: pd.DataFrame
    objective: float
    cost_eur: float
    emissions_kg: float
    soc_initial_kwh: float


def battery_limits(config: dict[str, Any]) -> dict[str, float]:
    """Battery limits in physical units (kWh and kW)."""
    battery = config["battery"]
    capacity = float(battery["capacity_kwh"])
    return {
        "soc_min_kwh": battery["soc_min_frac"] * capacity,
        "soc_max_kwh": battery["soc_max_frac"] * capacity,
        "soc_initial_kwh": battery["soc_initial_frac"] * capacity,
        # A battery with no capacity can neither charge nor discharge.
        "max_power_kw": float(battery["max_power_kw"]) if capacity > 0 else 0.0,
    }


def exchange_prices(price: pd.Series, config: dict[str, Any]) -> tuple[pd.Series, pd.Series]:
    """Buy and sell prices [EUR/kWh] derived from the day-ahead price."""
    buy = price + config["grid"]["import_surcharge_eur_kwh"]
    return buy.rename("buy_price_eur_kwh"), price.rename("sell_price_eur_kwh")


def build_model(
    load: np.ndarray,
    pv: np.ndarray,
    buy_price: np.ndarray,
    sell_price: np.ndarray,
    carbon: np.ndarray,
    config: dict[str, Any],
    soc_initial_kwh: float,
    carbon_weight: float,
) -> pyo.ConcreteModel:
    """Build the Pyomo model for one day (powers in kW, energies in kWh)."""
    battery, limits = config["battery"], battery_limits(config)
    step = float(config["time"]["step_hours"])
    grid_max = float(config["grid"]["max_power_kw"])
    power_max = limits["max_power_kw"]
    eta_charge, eta_discharge = battery["eta_charge"], battery["eta_discharge"]
    hours = len(load)

    model = pyo.ConcreteModel()
    # Pyomo sets are 1-based here, the input arrays 0-based: hence the `t - 1` below.
    model.T = pyo.RangeSet(1, hours)
    model.p_ch = pyo.Var(model.T, bounds=(0, power_max))
    model.p_dis = pyo.Var(model.T, bounds=(0, power_max))
    model.g_imp = pyo.Var(model.T, bounds=(0, grid_max))
    model.g_exp = pyo.Var(model.T, bounds=(0, grid_max))
    # (6) Cannot curtail more than what is produced.
    model.curt = pyo.Var(model.T, bounds=lambda _, t: (0, float(pv[t - 1])))
    # (3) State-of-charge limits.
    model.soc = pyo.Var(model.T, bounds=(limits["soc_min_kwh"], limits["soc_max_kwh"]))
    model.u = pyo.Var(model.T, domain=pyo.Binary)
    model.v = pyo.Var(model.T, domain=pyo.Binary)

    # (1) Power balance.
    model.balance = pyo.Constraint(
        model.T,
        rule=lambda m, t: pv[t - 1] - m.curt[t] + m.p_dis[t] + m.g_imp[t] == load[t - 1] + m.p_ch[t] + m.g_exp[t],
    )

    # (2) State-of-charge dynamics.
    def soc_dynamics(m: pyo.ConcreteModel, t: int) -> Any:
        previous = soc_initial_kwh if t == 1 else m.soc[t - 1]
        return m.soc[t] == previous + eta_charge * m.p_ch[t] * step - m.p_dis[t] * step / eta_discharge

    model.soc_dynamics = pyo.Constraint(model.T, rule=soc_dynamics)

    # (4) No charging and discharging in the same hour.
    model.charge_only = pyo.Constraint(model.T, rule=lambda m, t: m.p_ch[t] <= power_max * m.u[t])
    model.discharge_only = pyo.Constraint(model.T, rule=lambda m, t: m.p_dis[t] <= power_max * (1 - m.u[t]))
    # (5) No buying and selling in the same hour.
    model.import_only = pyo.Constraint(model.T, rule=lambda m, t: m.g_imp[t] <= grid_max * m.v[t])
    model.export_only = pyo.Constraint(model.T, rule=lambda m, t: m.g_exp[t] <= grid_max * (1 - m.v[t]))
    # (7) End of day: otherwise the model would empty the battery in the last hour.
    model.end_of_day = pyo.Constraint(expr=model.soc[hours] >= soc_initial_kwh)

    model.cost = pyo.Expression(
        expr=sum(buy_price[t - 1] * model.g_imp[t] - sell_price[t - 1] * model.g_exp[t] for t in model.T) * step
    )
    model.emissions = pyo.Expression(expr=sum(carbon[t - 1] * model.g_imp[t] for t in model.T) * step)
    model.objective = pyo.Objective(expr=model.cost + carbon_weight * model.emissions, sense=pyo.minimize)
    return model


def _check_inputs(series: dict[str, pd.Series], limits: dict[str, float], soc_initial_kwh: float) -> None:
    lengths = {len(values) for values in series.values()}
    if len(lengths) != 1 or lengths == {0}:
        raise ValueError("The input series must have the same, non-zero length.")
    for name, values in series.items():
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"Series `{name}` contains missing or non-finite values.")
    for name in ("load", "pv", "carbon"):
        if (series[name] < 0).any():
            raise ValueError(f"Series `{name}` contains negative values.")
    if not limits["soc_min_kwh"] <= soc_initial_kwh <= limits["soc_max_kwh"]:
        raise ValueError(
            f"Initial state of charge {soc_initial_kwh} kWh is outside the limits "
            f"[{limits['soc_min_kwh']}, {limits['soc_max_kwh']}]."
        )


def optimize_day(
    load: pd.Series,
    pv: pd.Series,
    price: pd.Series,
    carbon: pd.Series,
    config: dict[str, Any] | None = None,
    soc_initial_kwh: float | None = None,
    carbon_weight: float | None = None,
) -> Plan:
    """Compute the optimal battery and grid plan for the given hours.

    Inputs: forecast load and PV [kW], day-ahead price [EUR/kWh] and carbon
    intensity [kgCO2/kWh], all with the same index (23, 24 or 25 hours).
    `soc_initial_kwh` and `carbon_weight` (λ, EUR/kgCO2) default to the
    values in the configuration.

    Raises `ValueError` if the inputs are invalid and `OptimizationError` if
    the problem has no optimal solution.
    """
    config = config or load_config()
    limits = battery_limits(config)
    soc_initial_kwh = limits["soc_initial_kwh"] if soc_initial_kwh is None else float(soc_initial_kwh)
    carbon_weight = config["objective"]["carbon_weight_eur_kg"] if carbon_weight is None else float(carbon_weight)
    _check_inputs({"load": load, "pv": pv, "price": price, "carbon": carbon}, limits, soc_initial_kwh)
    buy_price, sell_price = exchange_prices(price, config)

    model = build_model(
        load.to_numpy(dtype=float),
        pv.to_numpy(dtype=float),
        buy_price.to_numpy(dtype=float),
        sell_price.to_numpy(dtype=float),
        carbon.to_numpy(dtype=float),
        config,
        soc_initial_kwh,
        carbon_weight,
    )
    settings = config["optimize"]
    solver = SolverFactory(settings["solver"])
    if not solver.available():
        raise OptimizationError(f"Solver `{settings['solver']}` is not available.")
    # Load the solution ourselves, only after checking that the solve succeeded.
    solver.config.load_solutions = False
    solver.config.rel_gap = settings["rel_gap"]
    solver.config.time_limit = settings["time_limit_s"]
    results = solver.solve(model, raise_exception_on_nonoptimal_result=False)
    if results.solution_status != SolutionStatus.optimal:
        raise OptimizationError(
            f"No optimal solution: {results.termination_condition.name} (day starting at {load.index[0]})."
        )
    results.solution_loader.load_vars()

    schedule = pd.DataFrame(
        {"load_kw": load.to_numpy(dtype=float), "pv_kw": pv.to_numpy(dtype=float)}, index=load.index
    )
    for column, variable in SCHEDULE_VARIABLES.items():
        values = np.array([pyo.value(getattr(model, variable)[t]) for t in model.T])
        # Wipe solver noise such as 1e-12, which would otherwise look like a tiny flow.
        schedule[column] = np.where(np.abs(values) < ZERO_TOLERANCE, 0.0, values)
    schedule["buy_price_eur_kwh"] = buy_price.to_numpy(dtype=float)
    schedule["sell_price_eur_kwh"] = sell_price.to_numpy(dtype=float)
    schedule["carbon_kg_kwh"] = carbon.to_numpy(dtype=float)
    return Plan(
        schedule=schedule,
        objective=float(pyo.value(model.objective)),
        cost_eur=float(pyo.value(model.cost)),
        emissions_kg=float(pyo.value(model.emissions)),
        soc_initial_kwh=soc_initial_kwh,
    )


def main() -> None:
    """Solve the example day and save its chart (`python -m microgrid.optimize`)."""
    # Local imports: only the example needs them, not the optimiser.
    from microgrid.dataset import load_dataset, local_days
    from microgrid.forecast import load_forecasts
    from microgrid.plotting import apply_style, plot_plan

    config = load_config()
    day = pd.Timestamp(config["optimize"]["example_day"])
    dataset, forecasts = load_dataset(config), load_forecasts(config)
    hours = forecasts.index[local_days(forecasts.index, config) == day]
    if hours.empty:
        raise SystemExit(f"No forecast available for {day.date()}.")
    plan = optimize_day(
        forecasts.loc[hours, "load_forecast"],
        forecasts.loc[hours, "pv_forecast"],
        dataset.loc[hours, "price_eur_kwh"],
        dataset.loc[hours, "carbon_kg_kwh"],
        config,
    )
    apply_style()
    figure = plot_plan(plan, config, title=f"Plan for {day.date()} (on forecast load and PV)")
    results_dir = resolve_path(config["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(results_dir / EXAMPLE_FILE, dpi=150, bbox_inches="tight")
    print(plan.schedule.round(2).to_string())
    print(f"planned cost {plan.cost_eur:.2f} EUR, planned emissions {plan.emissions_kg:.1f} kgCO2")


if __name__ == "__main__":
    main()
