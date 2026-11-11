"""Test della simulazione a orizzonte mobile su dati sintetici."""

import copy

import numpy as np
import pandas as pd
import pytest

from microgrid.config import load_config
from microgrid.dataset import local_days
from microgrid.optimize import battery_limits, optimize_day
from microgrid.simulate import SimulationError, apply_plan, simulate, simulate_rule

CONFIG = load_config()
TOLERANCE = 1e-6


def make_inputs(start: str, days: int, seed: int = 0, forecast_error: float = 0.0) -> pd.DataFrame:
    """Giorni locali interi con valori reali e previsioni (uguali ai reali se l'errore è nullo)."""
    timezone = CONFIG["site"]["timezone"]
    first = pd.Timestamp(start, tz=timezone)
    # La fine è una mezzanotte locale: con il cambio dell'ora non dista un multiplo di 24 ore.
    last = (pd.Timestamp(start) + pd.Timedelta(days=days)).tz_localize(timezone)
    index = pd.date_range(first.tz_convert("UTC"), last.tz_convert("UTC"), freq="h", inclusive="left")
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "load_kw": rng.uniform(2, 40, len(index)),
            "pv_kw": np.clip(rng.normal(15, 30, len(index)), 0, 90),
            "price_eur_kwh": rng.uniform(-0.05, 0.12, len(index)),
            "carbon_kg_kwh": rng.uniform(0.15, 0.60, len(index)),
        },
        index=index,
    )
    frame["load_forecast"] = np.clip(frame["load_kw"] + forecast_error * rng.normal(0, 1, len(index)), 0, None)
    frame["pv_forecast"] = np.clip(frame["pv_kw"] + forecast_error * rng.normal(0, 1, len(index)), 0, None)
    return frame


def one_day(load, pv, price=0.05, carbon=0.3) -> pd.DataFrame:
    index = pd.date_range("2017-06-01 00:00", periods=24, freq="h", tz="UTC")
    values = {"load_kw": load, "pv_kw": pv, "price_eur_kwh": price, "carbon_kg_kwh": carbon}
    return pd.DataFrame(
        {name: np.broadcast_to(np.asarray(v, dtype=float), 24) for name, v in values.items()}, index=index
    )


def zeros(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(0.0, index=frame.index)


def test_executing_a_plan_on_the_planned_values_reproduces_it():
    inputs = make_inputs("2017-06-01", 1)
    plan = optimize_day(inputs["load_kw"], inputs["pv_kw"], inputs["price_eur_kwh"], inputs["carbon_kg_kwh"], CONFIG)
    real = apply_plan(plan.schedule["charge_kw"], plan.schedule["discharge_kw"], inputs, CONFIG, plan.soc_initial_kwh)
    assert real["cost_eur"].sum() == pytest.approx(plan.cost_eur, abs=TOLERANCE)
    assert real["emissions_kg"].sum() == pytest.approx(plan.emissions_kg, abs=TOLERANCE)
    assert real["soc_kwh"].to_numpy() == pytest.approx(plan.schedule["soc_kwh"].to_numpy(), abs=TOLERANCE)


def test_forecast_error_is_absorbed_by_the_grid():
    actual = one_day(load=15.0, pv=0.0)
    # Piano fatto per un carico di 10 kW, senza usare la batteria: i 5 kW in più si acquistano.
    real = apply_plan(zeros(actual), zeros(actual), actual, CONFIG, 100.0)
    assert real["grid_import_kw"].to_numpy() == pytest.approx(15.0)
    assert real["soc_kwh"].to_numpy() == pytest.approx(100.0)
    buy = 0.05 + CONFIG["grid"]["import_surcharge_eur_kwh"]
    assert real["cost_eur"].sum() == pytest.approx(15.0 * 24 * buy)
    assert real["emissions_kg"].sum() == pytest.approx(15.0 * 24 * 0.3)


def test_battery_program_is_cut_at_the_soc_limits():
    actual = one_day(load=10.0, pv=0.0)
    limits, battery = battery_limits(CONFIG), CONFIG["battery"]
    power = pd.Series(battery["max_power_kw"], index=actual.index)
    charged = apply_plan(power, zeros(actual), actual, CONFIG, limits["soc_max_kwh"] - 9.5)
    # Restano 9,5 kWh di spazio: con rendimento 0,95 entrano 10 kW nella prima ora, poi nulla.
    assert charged["charge_kw"].iloc[0] == pytest.approx(9.5 / battery["eta_charge"])
    assert charged["charge_kw"].iloc[1:].sum() == pytest.approx(0, abs=TOLERANCE)
    assert charged["soc_kwh"].max() <= limits["soc_max_kwh"] + TOLERANCE
    discharged = apply_plan(zeros(actual), power, actual, CONFIG, limits["soc_min_kwh"])
    assert discharged["discharge_kw"].sum() == pytest.approx(0, abs=TOLERANCE)


def test_charge_is_reduced_when_import_would_exceed_grid_limit():
    config = copy.deepcopy(CONFIG)
    config["grid"]["max_power_kw"] = 40.0
    actual = one_day(load=30.0, pv=0.0)
    charge = zeros(actual)
    charge.iloc[0] = 50.0
    real = apply_plan(charge, zeros(actual), actual, config, 100.0)
    assert real["grid_import_kw"].iloc[0] == pytest.approx(40.0)
    assert real["charge_kw"].iloc[0] == pytest.approx(10.0)


def test_load_above_grid_limit_raises():
    config = copy.deepcopy(CONFIG)
    config["grid"]["max_power_kw"] = 40.0
    actual = one_day(load=60.0, pv=0.0)
    with pytest.raises(SimulationError):
        apply_plan(zeros(actual), zeros(actual), actual, config, 100.0)


def test_surplus_is_sold_or_curtailed_depending_on_price_and_grid_limit():
    price = np.full(24, 0.05)
    price[12:] = -0.02
    actual = one_day(load=10.0, pv=30.0, price=price)
    real = apply_plan(zeros(actual), zeros(actual), actual, CONFIG, 100.0)
    assert real["grid_export_kw"].iloc[:12].to_numpy() == pytest.approx(20.0)
    assert real["curtailment_kw"].iloc[:12].sum() == pytest.approx(0)
    assert real["grid_export_kw"].iloc[12:].sum() == pytest.approx(0)
    assert real["curtailment_kw"].iloc[12:].to_numpy() == pytest.approx(20.0)

    config = copy.deepcopy(CONFIG)
    config["grid"]["max_power_kw"] = 15.0
    limited = apply_plan(zeros(actual), zeros(actual), actual, config, 100.0)
    assert limited["grid_export_kw"].iloc[:12].to_numpy() == pytest.approx(15.0)
    assert limited["curtailment_kw"].iloc[:12].to_numpy() == pytest.approx(5.0)


def test_simulation_carries_soc_from_one_day_to_the_next():
    # Fine ottobre 2017: il 29 ha 25 ore.
    inputs = make_inputs("2017-10-27", 5, seed=1, forecast_error=5.0)
    hourly, daily = simulate(inputs, CONFIG)
    limits = battery_limits(CONFIG)
    assert daily["hours"].tolist() == [24, 24, 25, 24, 24]
    assert len(hourly) == len(inputs)
    assert daily["soc_start_kwh"].iloc[0] == limits["soc_initial_kwh"]
    assert daily["soc_start_kwh"].iloc[1:].to_numpy() == pytest.approx(daily["soc_end_kwh"].iloc[:-1].to_numpy())

    # Lo stato di carica è continuo anche da un'ora all'altra, su tutto il periodo.
    battery = CONFIG["battery"]
    previous = np.append(limits["soc_initial_kwh"], hourly["soc_kwh"].to_numpy()[:-1])
    expected = (
        previous + battery["eta_charge"] * hourly["charge_kw"] - hourly["discharge_kw"] / battery["eta_discharge"]
    )
    assert hourly["soc_kwh"].to_numpy() == pytest.approx(expected.to_numpy(), abs=TOLERANCE)
    assert hourly["soc_kwh"].between(limits["soc_min_kwh"] - TOLERANCE, limits["soc_max_kwh"] + TOLERANCE).all()

    # Bilancio di potenza sui flussi reali e coerenza fra serie oraria e giornaliera.
    supply = hourly["pv_kw"] - hourly["curtailment_kw"] + hourly["discharge_kw"] + hourly["grid_import_kw"]
    demand = hourly["load_kw"] + hourly["charge_kw"] + hourly["grid_export_kw"]
    assert (supply - demand).abs().max() < TOLERANCE
    days = local_days(hourly.index, CONFIG)
    assert daily["cost_eur"].to_numpy() == pytest.approx(hourly["cost_eur"].groupby(days.to_numpy()).sum().to_numpy())
    assert daily["emissions_kg"].sum() == pytest.approx(hourly["emissions_kg"].sum())


def test_perfect_forecast_costs_what_was_planned_and_never_more_than_forecast():
    inputs = make_inputs("2017-06-01", 4, seed=2, forecast_error=8.0)
    _, with_forecast = simulate(inputs, CONFIG)
    _, perfect = simulate(inputs, CONFIG, load_column="load_kw", pv_column="pv_kw")
    assert perfect["cost_eur"].to_numpy() == pytest.approx(perfect["planned_cost_eur"].to_numpy(), abs=TOLERANCE)
    assert perfect["cost_eur"].sum() <= with_forecast["cost_eur"].sum() + TOLERANCE


def test_simulation_rejects_missing_values():
    inputs = make_inputs("2017-06-01", 1)
    inputs.loc[inputs.index[5], "pv_forecast"] = np.nan
    with pytest.raises(ValueError):
        simulate(inputs, CONFIG)


def test_naive_rule_charges_on_surplus_and_discharges_on_deficit():
    pv = np.where(np.arange(24) < 12, 30.0, 0.0)
    actual = one_day(load=10.0, pv=pv)
    limits = battery_limits(CONFIG)
    hourly, daily = simulate_rule(actual, CONFIG)
    # Mattina: 20 kW di surplus vanno in batteria finché c'è spazio, poi alla rete.
    assert hourly["charge_kw"].iloc[0] == pytest.approx(20.0)
    assert hourly["grid_import_kw"].iloc[:12].sum() == pytest.approx(0)
    assert hourly["soc_kwh"].max() == pytest.approx(limits["soc_max_kwh"])
    assert hourly["grid_export_kw"].iloc[11] == pytest.approx(20.0)
    # Sera: la batteria copre il carico finché non è al minimo, poi si acquista.
    assert hourly["discharge_kw"].iloc[12] == pytest.approx(10.0)
    assert hourly["grid_import_kw"].iloc[12] == pytest.approx(0)
    assert hourly["soc_kwh"].min() >= limits["soc_min_kwh"] - TOLERANCE
    assert daily["cost_eur"].sum() == pytest.approx(hourly["cost_eur"].sum())


def test_rule_without_battery_only_exchanges_with_the_grid():
    actual = one_day(load=10.0, pv=np.where(np.arange(24) < 12, 30.0, 0.0))
    hourly, daily = simulate_rule(actual, CONFIG, use_battery=False)
    assert hourly[["charge_kw", "discharge_kw"]].sum().sum() == pytest.approx(0)
    assert hourly["grid_export_kw"].iloc[:12].to_numpy() == pytest.approx(20.0)
    assert hourly["grid_import_kw"].iloc[12:].to_numpy() == pytest.approx(10.0)
    assert daily["soc_end_kwh"].iloc[-1] == battery_limits(CONFIG)["soc_initial_kwh"]


def planned_day(load, pv, charge=0.0, discharge=0.0) -> pd.DataFrame:
    """Piano costante su 24 ore, con lo scambio di rete che chiude il bilancio."""
    plan = one_day(load=load, pv=pv)[["load_kw", "pv_kw"]]
    plan["charge_kw"], plan["discharge_kw"] = charge, discharge
    net = plan["load_kw"] - plan["pv_kw"] + plan["charge_kw"] - plan["discharge_kw"]
    plan["grid_import_kw"], plan["grid_export_kw"] = net.clip(lower=0), (-net).clip(lower=0)
    return plan


def test_correction_lets_the_battery_absorb_the_forecast_error():
    # Piano: carico previsto 10 kW, coperto dalla batteria, nessuno scambio con la rete.
    plan = planned_day(load=10.0, pv=0.0, discharge=10.0)
    actual = one_day(load=14.0, pv=0.0)
    literal = apply_plan(plan["charge_kw"], plan["discharge_kw"], actual, CONFIG, 180.0)
    corrected = apply_plan(plan["charge_kw"], plan["discharge_kw"], actual, CONFIG, 180.0, plan)
    # Alla lettera i 4 kW in più si acquistano; con la correzione li copre la batteria.
    assert literal["grid_import_kw"].iloc[0] == pytest.approx(4.0)
    assert corrected["discharge_kw"].iloc[0] == pytest.approx(14.0)
    assert corrected["grid_import_kw"].iloc[0] == pytest.approx(0)


def test_correction_never_sells_energy_stored_for_self_consumption():
    # Il carico reale è più basso del previsto: la batteria scarica solo quanto serve.
    plan = planned_day(load=10.0, pv=0.0, discharge=10.0)
    actual = one_day(load=4.0, pv=0.0)
    literal = apply_plan(plan["charge_kw"], plan["discharge_kw"], actual, CONFIG, 180.0)
    corrected = apply_plan(plan["charge_kw"], plan["discharge_kw"], actual, CONFIG, 180.0, plan)
    assert literal["grid_export_kw"].iloc[0] == pytest.approx(6.0)
    assert corrected["discharge_kw"].iloc[0] == pytest.approx(4.0)
    assert corrected["grid_export_kw"].iloc[0] == pytest.approx(0)


def test_correction_never_buys_energy_the_plan_expected_from_the_sun():
    # Piano: 20 kW di surplus previsto in batteria. Il sole reale è meno del carico.
    plan = planned_day(load=10.0, pv=30.0, charge=20.0)
    actual = one_day(load=10.0, pv=5.0)
    literal = apply_plan(plan["charge_kw"], plan["discharge_kw"], actual, CONFIG, 100.0)
    corrected = apply_plan(plan["charge_kw"], plan["discharge_kw"], actual, CONFIG, 100.0, plan)
    assert literal["grid_import_kw"].iloc[0] == pytest.approx(25.0)
    assert corrected["charge_kw"].iloc[0] == pytest.approx(0)
    # Il deficit reale di 5 kW è coperto dalla batteria, come vuole il piano (scambio nullo).
    assert corrected["discharge_kw"].iloc[0] == pytest.approx(5.0)
    assert corrected["grid_import_kw"].iloc[0] == pytest.approx(0)


def test_correction_keeps_planned_exchanges_with_the_grid():
    # Piano: caricare 30 kW dalla rete di notte (nessun sole) e acquistare anche il carico.
    plan = planned_day(load=10.0, pv=0.0, charge=30.0)
    actual = one_day(load=10.0, pv=0.0)
    corrected = apply_plan(plan["charge_kw"], plan["discharge_kw"], actual, CONFIG, 100.0, plan)
    assert corrected["charge_kw"].iloc[0] == pytest.approx(30.0)
    assert corrected["grid_import_kw"].iloc[0] == pytest.approx(40.0)


def test_correction_changes_nothing_when_the_forecast_is_exact():
    # Prezzi positivi: il piano non taglia fotovoltaico.
    inputs = make_inputs("2017-06-01", 3, seed=4)
    inputs["price_eur_kwh"] = inputs["price_eur_kwh"].abs() + 0.01
    _, literal = simulate(inputs, CONFIG, load_column="load_kw", pv_column="pv_kw")
    _, corrected = simulate(inputs, CONFIG, load_column="load_kw", pv_column="pv_kw", correct=True)
    assert corrected["cost_eur"].to_numpy() == pytest.approx(literal["cost_eur"].to_numpy(), abs=1e-5)
