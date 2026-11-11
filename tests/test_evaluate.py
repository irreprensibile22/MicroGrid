"""Test dei confronti fra strategie su pochi giorni sintetici."""

import numpy as np
import pandas as pd
import pytest

from microgrid.config import load_config
from microgrid.evaluate import STRATEGIES, add_savings, compare, forecast_errors, pareto, run_strategy, with_parameter

CONFIG = load_config()


def make_inputs(days: int = 3, seed: int = 0) -> pd.DataFrame:
    timezone = CONFIG["site"]["timezone"]
    first = pd.Timestamp("2017-06-01", tz=timezone)
    last = pd.Timestamp("2017-06-01", tz=timezone) + pd.Timedelta(days=days)
    index = pd.date_range(first.tz_convert("UTC"), last.tz_convert("UTC"), freq="h", inclusive="left")
    rng = np.random.default_rng(seed)
    hour = index.tz_convert(timezone).hour
    frame = pd.DataFrame(
        {
            "load_kw": rng.uniform(5, 25, len(index)),
            "pv_kw": np.clip(60 * np.sin(np.pi * (hour - 6) / 12), 0, None) * rng.uniform(0.3, 1.0, len(index)),
            "price_eur_kwh": rng.uniform(0.01, 0.08, len(index)),
            "carbon_kg_kwh": rng.uniform(0.15, 0.60, len(index)),
        },
        index=index,
    )
    frame["load_forecast"] = np.clip(frame["load_kw"] + rng.normal(0, 4, len(index)), 0, None)
    frame["pv_forecast"] = np.clip(frame["pv_kw"] + rng.normal(0, 8, len(index)), 0, None)
    frame["load_naive"] = frame["load_kw"].mean()
    return frame


def test_every_strategy_runs_and_respects_the_power_balance():
    inputs = make_inputs()
    for name in (*STRATEGIES, "milp_corrected_naive_load"):
        hourly, daily = run_strategy(name, inputs, CONFIG)
        supply = hourly["pv_kw"] - hourly["curtailment_kw"] + hourly["discharge_kw"] + hourly["grid_import_kw"]
        demand = hourly["load_kw"] + hourly["charge_kw"] + hourly["grid_export_kw"]
        assert (supply - demand).abs().max() < 1e-6, name
        assert len(daily) == 3, name
    with pytest.raises(ValueError):
        run_strategy("sconosciuta", inputs, CONFIG)


def test_comparison_table_is_consistent():
    inputs = make_inputs()
    table, daily_costs = compare(inputs, CONFIG)
    assert list(table.index) == list(STRATEGIES)
    assert table.loc["no_battery", "cost_saving_eur"] == 0
    assert table.loc["no_battery", "battery_cycles"] == 0
    assert table.loc["perfect", "saving_captured"] == pytest.approx(1.0)
    # Con previsione perfetta il MILP non costa mai più che con la previsione vera.
    assert table.loc["perfect", "cost_eur"] <= table.loc["milp", "cost_eur"] + 1e-6
    assert daily_costs.sum().to_numpy() == pytest.approx(table["cost_eur"].to_numpy())


def test_add_savings_values():
    table = pd.DataFrame(
        {"cost_eur": [100.0, 80.0, 60.0], "emissions_kg": [50.0, 45.0, 40.0]}, index=["no_battery", "milp", "perfect"]
    )
    result = add_savings(table)
    assert result.loc["milp", "cost_saving_pct"] == pytest.approx(20.0)
    assert result.loc["milp", "emissions_saving_pct"] == pytest.approx(10.0)
    assert result.loc["milp", "saving_captured"] == pytest.approx(0.5)


def test_forecast_errors_are_daily():
    inputs = make_inputs()
    errors = forecast_errors(inputs, CONFIG)
    assert len(errors) == 3
    expected = (inputs["load_forecast"] - inputs["load_kw"]).abs().iloc[:24].mean()
    assert errors["load_mae_kw"].iloc[0] == pytest.approx(expected)


def test_with_parameter_scales_pv_and_leaves_the_original_untouched():
    inputs = make_inputs()
    scaled, changed = with_parameter(inputs, CONFIG, "pv_peak_kw", 2 * CONFIG["pv"]["peak_kw"])
    assert scaled["pv_kw"].to_numpy() == pytest.approx(2 * inputs["pv_kw"].to_numpy())
    assert scaled["pv_forecast"].to_numpy() == pytest.approx(2 * inputs["pv_forecast"].to_numpy())
    assert changed["pv"]["peak_kw"] == 2 * CONFIG["pv"]["peak_kw"]
    same, changed = with_parameter(inputs, CONFIG, "battery_capacity_kwh", 50.0)
    assert same is inputs and changed["battery"]["capacity_kwh"] == 50.0
    assert CONFIG["battery"]["capacity_kwh"] != 50.0


def test_higher_carbon_weight_does_not_increase_emissions_with_perfect_forecast():
    inputs = make_inputs()
    config = {**CONFIG, "evaluate": {**CONFIG["evaluate"], "pareto_carbon_weights": [0.0, 2.0]}}
    front = pareto(inputs, config, strategies=("perfect",)).set_index("carbon_weight_eur_kg")
    assert front.loc[2.0, "emissions_kg"] <= front.loc[0.0, "emissions_kg"] + 1e-6
    assert front.loc[2.0, "cost_eur"] >= front.loc[0.0, "cost_eur"] - 1e-6
