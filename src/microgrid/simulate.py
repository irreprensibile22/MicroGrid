"""Rolling-horizon simulation: one plan per day, executed on the actual data.

Usage:

    python -m microgrid.simulate

For each day of the simulated period:

1. the MILP computes the battery plan on forecast load and PV;
2. the plan is executed on the actual values;
3. the energy left in the battery is the initial state of the next day.

Writes the hourly detail to `data/processed/simulation.parquet` and the daily
cost and emission series to `results/simulation_daily.csv`.

A plan can be executed in two ways (`apply_plan`).

To the letter: the battery follows its schedule and the grid absorbs the
difference between forecast and reality.

With real-time correction (`correct=True`): we track the grid exchange the
plan expected instead of the battery schedule, so the battery absorbs the
forecast error for as long as it can. Two caps avoid bad trades:

- the battery charges at most the actual surplus, plus whatever grid
  charging the plan had decided on;
- the battery discharges at most the actual deficit, plus whatever
  discharging to the grid the plan had decided on.

In both cases the battery stays within its state-of-charge limits, charging
is reduced if imports would exceed the grid limit, and PV is curtailed when
the sell price is negative or exports would exceed the grid limit.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from microgrid.config import load_config, resolve_path
from microgrid.dataset import load_dataset, local_days
from microgrid.forecast import load_forecasts
from microgrid.optimize import ZERO_TOLERANCE, battery_limits, exchange_prices, optimize_day

HOURLY_FILE = "simulation.parquet"
DAILY_FILE = "simulation_daily.csv"
INPUT_COLUMNS = ["load_kw", "pv_kw", "price_eur_kwh", "carbon_kg_kwh"]


class SimulationError(RuntimeError):
    """The actual load cannot be met within the limits of the plant."""


def apply_plan(
    charge: pd.Series,
    discharge: pd.Series,
    actual: pd.DataFrame,
    config: dict[str, Any],
    soc_initial_kwh: float,
    schedule: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Execute a battery schedule on the actual values of a period.

    `charge` and `discharge` are the scheduled powers [kW]; `actual` holds the
    real `load_kw`, `pv_kw`, `price_eur_kwh` and `carbon_kg_kwh`, with the
    same index. Returns, for each hour, the flows that really happened, the
    energy in the battery at the end of the hour, the cost [EUR] and the
    emissions [kgCO2].

    If the full plan is given (`schedule`, as returned by `optimize_day`),
    the real-time correction described in the module docstring is applied.
    """
    battery, limits = config["battery"], battery_limits(config)
    step = float(config["time"]["step_hours"])
    grid_max = float(config["grid"]["max_power_kw"])
    eta_charge, eta_discharge = battery["eta_charge"], battery["eta_discharge"]
    power_max = limits["max_power_kw"]
    buy_price, sell_price = exchange_prices(actual["price_eur_kwh"], config)

    soc = float(soc_initial_kwh)
    rows = []
    for hour in range(len(actual)):
        load, pv = float(actual["load_kw"].iloc[hour]), float(actual["pv_kw"].iloc[hour])
        charge_kw, discharge_kw = float(charge.iloc[hour]), float(discharge.iloc[hour])
        if schedule is not None:
            planned = schedule.iloc[hour]
            planned_surplus = float(planned["pv_kw"] - planned["load_kw"])
            # Parts of the plan that trade with the grid on purpose (price arbitrage).
            from_grid = charge_kw - min(charge_kw, max(planned_surplus, 0.0))
            to_grid = discharge_kw - min(discharge_kw, max(-planned_surplus, 0.0))
            # Power the battery must deliver (+) or absorb (−) so that the grid
            # exchange matches what the plan expected.
            planned_exchange = float(planned["grid_import_kw"] - planned["grid_export_kw"])
            needed = (load - pv) - planned_exchange
            discharge_kw = min(max(needed, 0.0), max(load - pv, 0.0) + to_grid, power_max)
            charge_kw = min(max(-needed, 0.0), max(pv - load, 0.0) + from_grid, power_max)
        # Whatever the schedule says, the battery cannot leave its state-of-charge limits.
        charge_kw = min(charge_kw, max(limits["soc_max_kwh"] - soc, 0.0) / (eta_charge * step))
        discharge_kw = min(discharge_kw, max(soc - limits["soc_min_kwh"], 0.0) * eta_discharge / step)

        # The grid takes whatever is left over.
        net = load - pv + charge_kw - discharge_kw
        if net > grid_max:
            charge_kw = max(charge_kw - (net - grid_max), 0.0)
            net = load - pv + charge_kw - discharge_kw
            if net > grid_max + ZERO_TOLERANCE:
                raise SimulationError(
                    f"At {actual.index[hour]} the grid would have to supply {net:.1f} kW, "
                    f"above its limit of {grid_max:.1f} kW."
                )
        grid_import = max(net, 0.0)
        surplus = max(-net, 0.0)
        # Selling at a negative price costs money: curtail PV instead, as far as needed.
        curtailment = min(surplus, pv) if sell_price.iloc[hour] < 0 else 0.0
        curtailment = max(curtailment, min(surplus - grid_max, pv))
        grid_export = surplus - curtailment

        soc = soc + eta_charge * charge_kw * step - discharge_kw * step / eta_discharge
        rows.append(
            {
                "charge_kw": charge_kw,
                "discharge_kw": discharge_kw,
                "grid_import_kw": grid_import,
                "grid_export_kw": grid_export,
                "curtailment_kw": curtailment,
                "soc_kwh": soc,
                "cost_eur": (buy_price.iloc[hour] * grid_import - sell_price.iloc[hour] * grid_export) * step,
                "emissions_kg": float(actual["carbon_kg_kwh"].iloc[hour]) * grid_import * step,
            }
        )
    return pd.DataFrame(rows, index=actual.index)


def daily_totals(actual: pd.DataFrame, real: pd.DataFrame, soc_start_kwh: float, step: float) -> dict[str, Any]:
    """Totals of one day from the actual values and the flows that happened."""
    return {
        "hours": len(actual),
        "cost_eur": real["cost_eur"].sum(),
        "emissions_kg": real["emissions_kg"].sum(),
        "load_kwh": actual["load_kw"].sum() * step,
        "pv_kwh": actual["pv_kw"].sum() * step,
        "import_kwh": real["grid_import_kw"].sum() * step,
        "export_kwh": real["grid_export_kw"].sum() * step,
        "curtailed_kwh": real["curtailment_kw"].sum() * step,
        "battery_discharge_kwh": real["discharge_kw"].sum() * step,
        "soc_start_kwh": soc_start_kwh,
        "soc_end_kwh": real["soc_kwh"].iloc[-1],
    }


def simulate_rule(
    inputs: pd.DataFrame, config: dict[str, Any], use_battery: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate the strategies that use neither forecasts nor optimisation.

    With `use_battery=True` this is the naive rule: the battery charges when
    PV exceeds the load and discharges when it falls short, within its
    limits. With `use_battery=False` the battery is never used. Returns the
    hourly detail and the daily series, like `simulate`.
    """
    if inputs[INPUT_COLUMNS].isna().any().any():
        raise ValueError("The simulation inputs contain missing values.")
    step = float(config["time"]["step_hours"])
    limits = battery_limits(config)
    power = limits["max_power_kw"] if use_battery else 0.0
    surplus = inputs["pv_kw"] - inputs["load_kw"]
    soc = limits["soc_initial_kwh"]
    # Ask for the whole surplus or deficit; `apply_plan` trims it to what the battery can take.
    real = apply_plan(surplus.clip(0, power), (-surplus).clip(0, power), inputs, config, soc)
    days = local_days(inputs.index, config)
    daily = []
    for day in days.unique():
        of_day = days == day
        daily.append({"day": day.date().isoformat(), **daily_totals(inputs[of_day], real[of_day], soc, step)})
        soc = float(real["soc_kwh"][of_day].iloc[-1])
    return pd.concat([inputs[INPUT_COLUMNS], real], axis=1), pd.DataFrame(daily).set_index("day")


def simulate(
    inputs: pd.DataFrame,
    config: dict[str, Any],
    load_column: str = "load_forecast",
    pv_column: str = "pv_forecast",
    carbon_weight: float | None = None,
    correct: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate day by day the hours in `inputs`, which must be consecutive.

    `inputs` holds the actual values (`load_kw`, `pv_kw`, `price_eur_kwh`,
    `carbon_kg_kwh`) and the columns `load_column` and `pv_column` the MILP
    plans on. Passing `load_kw` and `pv_kw` as planning columns too gives the
    perfect-forecast case.

    With `correct=True` the plan is executed with the real-time correction
    described in the module docstring.

    Returns the hourly detail and the daily series. In the hourly detail the
    `planned_*` columns are the schedule, the others what actually happened.
    """
    needed = [*INPUT_COLUMNS, load_column, pv_column]
    if inputs[needed].isna().any().any():
        raise ValueError("The simulation inputs contain missing values.")
    step = float(config["time"]["step_hours"])
    days = local_days(inputs.index, config)
    soc = battery_limits(config)["soc_initial_kwh"]
    hourly, daily = [], []
    for day in days.unique():
        actual = inputs[days == day]
        plan = optimize_day(
            actual[load_column],
            actual[pv_column],
            actual["price_eur_kwh"],
            actual["carbon_kg_kwh"],
            config,
            soc_initial_kwh=soc,
            carbon_weight=carbon_weight,
        )
        schedule = plan.schedule
        real = apply_plan(
            schedule["charge_kw"], schedule["discharge_kw"], actual, config, soc, schedule if correct else None
        )
        hourly.append(
            pd.concat(
                [
                    actual[["load_kw", "pv_kw", "price_eur_kwh", "carbon_kg_kwh"]],
                    pd.DataFrame(
                        {
                            "planned_load_kw": schedule["load_kw"],
                            "planned_pv_kw": schedule["pv_kw"],
                            "planned_charge_kw": schedule["charge_kw"],
                            "planned_discharge_kw": schedule["discharge_kw"],
                            "planned_soc_kwh": schedule["soc_kwh"],
                        }
                    ),
                    real,
                ],
                axis=1,
            )
        )
        daily.append(
            {
                "day": day.date().isoformat(),
                **daily_totals(actual, real, soc, step),
                "planned_cost_eur": plan.cost_eur,
                "planned_emissions_kg": plan.emissions_kg,
            }
        )
        # Tomorrow starts from where the battery really ended up, not from where the plan
        # thought it would. The clip only guards against floating-point drift at the limits.
        soc = float(np.clip(real["soc_kwh"].iloc[-1], *_soc_bounds(config)))
    return pd.concat(hourly), pd.DataFrame(daily).set_index("day")


def _soc_bounds(config: dict[str, Any]) -> tuple[float, float]:
    limits = battery_limits(config)
    return limits["soc_min_kwh"], limits["soc_max_kwh"]


def simulation_inputs(config: dict[str, Any], split: str) -> pd.DataFrame:
    """Actual values and forecasts for the hours of `split` (`validation` or `test`)."""
    dataset, forecasts = load_dataset(config), load_forecasts(config)
    hours = forecasts.index[forecasts["split"] == split]
    return pd.concat(
        [dataset.loc[hours, INPUT_COLUMNS], forecasts.loc[hours, ["load_forecast", "pv_forecast"]]], axis=1
    )


def load_simulation(config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Read the hourly detail saved by `python -m microgrid.simulate`."""
    config = config or load_config()
    path = resolve_path(config["paths"]["processed"]) / HOURLY_FILE
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}: run `python -m microgrid.simulate` first.")
    return pd.read_parquet(path)


def main() -> None:
    """Simulate the configured period and save the results (`python -m microgrid.simulate`)."""
    config = load_config()
    split = config["simulate"]["split"]
    hourly, daily = simulate(simulation_inputs(config, split), config)
    results_dir = resolve_path(config["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(resolve_path(config["paths"]["processed"]) / HOURLY_FILE)
    daily.to_csv(results_dir / DAILY_FILE, float_format="%.4f")
    print(daily.round(2).to_string())
    print(
        f"\n{split}: {len(daily)} days, cost {daily['cost_eur'].sum():.2f} EUR "
        f"(planned {daily['planned_cost_eur'].sum():.2f}), emissions {daily['emissions_kg'].sum():.1f} kgCO2"
    )


if __name__ == "__main__":
    main()
