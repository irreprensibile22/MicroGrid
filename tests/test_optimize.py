"""Test dell'ottimizzatore su giorni sintetici, senza file di dati."""

import copy

import numpy as np
import pandas as pd
import pytest

from microgrid.config import load_config
from microgrid.optimize import OptimizationError, battery_limits, optimize_day

CONFIG = load_config()
TOLERANCE = 1e-6


def hours(periods: int = 24) -> pd.DatetimeIndex:
    return pd.date_range("2017-06-01 00:00", periods=periods, freq="h", tz="UTC")


def series(values, periods: int = 24) -> pd.Series:
    return pd.Series(np.broadcast_to(np.asarray(values, dtype=float), periods).copy(), index=hours(periods))


def with_settings(**sections) -> dict:
    """Copia della configurazione con alcune chiavi sostituite, sezione per sezione."""
    config = copy.deepcopy(CONFIG)
    for section, values in sections.items():
        config[section].update(values)
    return config


def random_day(seed: int, periods: int = 24) -> dict[str, pd.Series]:
    """Giorno casuale con surplus e deficit, e prezzi anche negativi."""
    rng = np.random.default_rng(seed)
    return {
        "load": series(rng.uniform(2, 40, periods), periods),
        "pv": series(np.clip(rng.normal(20, 30, periods), 0, 90), periods),
        "price": series(rng.uniform(-0.10, 0.12, periods), periods),
        "carbon": series(rng.uniform(0.15, 0.60, periods), periods),
    }


def test_charges_when_cheap_and_discharges_when_expensive():
    # Nessun fotovoltaico, carico costante, prezzo basso di notte e alto di sera.
    price = np.full(24, 0.05)
    price[:6] = 0.02
    price[18:22] = 0.10
    load = series(10.0)
    plan = optimize_day(load, series(0.0), series(price), series(0.3), CONFIG)
    schedule = plan.schedule
    assert schedule["charge_kw"].iloc[:6].sum() > 0
    assert schedule["charge_kw"].iloc[6:].sum() == pytest.approx(0, abs=TOLERANCE)
    assert schedule["discharge_kw"].iloc[:6].sum() == pytest.approx(0, abs=TOLERANCE)
    # Nelle ore di punta la batteria copre tutto il carico.
    assert schedule["grid_import_kw"].iloc[18:22].to_numpy() == pytest.approx(0, abs=TOLERANCE)

    # Soluzione attesa, calcolata a mano. La batteria parte da metà: di notte si
    # riempie fino al massimo, poi cede la stessa energia e torna al livello iniziale.
    battery, limits = CONFIG["battery"], battery_limits(CONFIG)
    surcharge = CONFIG["grid"]["import_surcharge_eur_kwh"]
    stored = limits["soc_max_kwh"] - limits["soc_initial_kwh"]
    bought = stored / battery["eta_charge"]
    delivered = stored * battery["eta_discharge"]
    peak_energy = 10.0 * 4
    without_battery = 10.0 * (price + surcharge).sum()
    expected = (
        without_battery
        + bought * (0.02 + surcharge)
        - peak_energy * (0.10 + surcharge)
        - (delivered - peak_energy) * (0.05 + surcharge)
    )
    assert plan.cost_eur == pytest.approx(expected, rel=1e-6)
    assert plan.cost_eur < without_battery


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("surcharge", [0.20, 0.0])
@pytest.mark.parametrize("periods", [23, 24, 25])
def test_plan_respects_all_constraints(seed, surcharge, periods):
    # Con oneri nulli e prezzi negativi acquistare è pagato: è il caso in cui
    # le variabili binarie sono indispensabili.
    config = with_settings(grid={"import_surcharge_eur_kwh": surcharge})
    day = random_day(seed, periods)
    plan = optimize_day(**day, config=config)
    s, battery, limits = plan.schedule, config["battery"], battery_limits(config)
    assert len(s) == periods

    # Bilancio di potenza in ogni ora.
    supply = s["pv_kw"] - s["curtailment_kw"] + s["discharge_kw"] + s["grid_import_kw"]
    demand = s["load_kw"] + s["charge_kw"] + s["grid_export_kw"]
    assert (supply - demand).abs().max() < TOLERANCE

    # Dinamica e limiti dello stato di carica.
    previous = np.append(plan.soc_initial_kwh, s["soc_kwh"].to_numpy()[:-1])
    expected_soc = previous + battery["eta_charge"] * s["charge_kw"] - s["discharge_kw"] / battery["eta_discharge"]
    assert (s["soc_kwh"] - expected_soc).abs().max() < TOLERANCE
    assert s["soc_kwh"].min() >= limits["soc_min_kwh"] - TOLERANCE
    assert s["soc_kwh"].max() <= limits["soc_max_kwh"] + TOLERANCE
    assert s["soc_kwh"].iloc[-1] >= plan.soc_initial_kwh - TOLERANCE

    # Limiti di potenza, mai due operazioni opposte nella stessa ora.
    assert s[["charge_kw", "discharge_kw"]].max().max() <= battery["max_power_kw"] + TOLERANCE
    assert s[["grid_import_kw", "grid_export_kw"]].max().max() <= config["grid"]["max_power_kw"] + TOLERANCE
    assert s[["charge_kw", "discharge_kw"]].min(axis=1).max() < TOLERANCE
    assert s[["grid_import_kw", "grid_export_kw"]].min(axis=1).max() < TOLERANCE
    assert (s["curtailment_kw"] <= s["pv_kw"] + TOLERANCE).all()
    assert (s[["charge_kw", "discharge_kw", "grid_import_kw", "grid_export_kw", "curtailment_kw"]] >= 0).all().all()

    # Il costo dichiarato è quello degli scambi con la rete.
    cost = (s["buy_price_eur_kwh"] * s["grid_import_kw"] - s["sell_price_eur_kwh"] * s["grid_export_kw"]).sum()
    assert plan.cost_eur == pytest.approx(cost, abs=TOLERANCE)


def test_zero_capacity_battery_matches_no_battery():
    config = with_settings(battery={"capacity_kwh": 0.0})
    rng = np.random.default_rng(1)
    load, pv = series(rng.uniform(2, 40, 24)), series(rng.uniform(0, 80, 24))
    # Prezzi positivi: il surplus si vende sempre, non si taglia.
    plan = optimize_day(load, pv, series(rng.uniform(0.01, 0.10, 24)), series(0.3), config)
    s = plan.schedule
    net = (load - pv).to_numpy()
    assert s["grid_import_kw"].to_numpy() == pytest.approx(np.clip(net, 0, None), abs=TOLERANCE)
    assert s["grid_export_kw"].to_numpy() == pytest.approx(np.clip(-net, 0, None), abs=TOLERANCE)
    assert s[["charge_kw", "discharge_kw", "curtailment_kw", "soc_kwh"]].abs().max().max() < TOLERANCE


def test_battery_never_does_worse_than_no_battery():
    day = random_day(3)
    with_battery = optimize_day(**day, config=CONFIG)
    without = optimize_day(**day, config=with_settings(battery={"capacity_kwh": 0.0}))
    assert with_battery.cost_eur <= without.cost_eur + TOLERANCE


def test_surplus_is_curtailed_when_selling_costs_money():
    # Batteria piena, surplus di 20 kW e prezzo di vendita negativo: conviene tagliare.
    limits = battery_limits(CONFIG)
    plan = optimize_day(
        series(10.0), series(30.0), series(-0.05), series(0.3), CONFIG, soc_initial_kwh=limits["soc_max_kwh"]
    )
    assert plan.schedule["grid_export_kw"].sum() == pytest.approx(0, abs=TOLERANCE)
    assert plan.schedule["grid_import_kw"].sum() == pytest.approx(0, abs=TOLERANCE)
    assert plan.cost_eur == pytest.approx(0, abs=TOLERANCE)
    # Quanto si taglia in ogni ora non è univoco: la batteria può muoversi a costo nullo.
    assert plan.schedule["curtailment_kw"].sum() > 0


def test_carbon_weight_moves_purchases_to_clean_hours():
    # Prezzo costante; la rete è pulita nella prima metà del giorno, sporca nella seconda.
    carbon = np.where(np.arange(24) < 12, 0.1, 0.6)
    day = {"load": series(10.0), "pv": series(0.0), "price": series(0.05), "carbon": series(carbon)}
    cost_only = optimize_day(**day, config=CONFIG, carbon_weight=0.0)
    low_carbon = optimize_day(**day, config=CONFIG, carbon_weight=1.0)
    assert low_carbon.emissions_kg < cost_only.emissions_kg - 1.0
    assert low_carbon.cost_eur >= cost_only.cost_eur - TOLERANCE
    assert low_carbon.schedule["charge_kw"].iloc[:12].sum() > 0
    assert low_carbon.objective == pytest.approx(low_carbon.cost_eur + 1.0 * low_carbon.emissions_kg)


def test_initial_soc_is_passed_from_outside():
    limits = battery_limits(CONFIG)
    plan = optimize_day(**random_day(0), config=CONFIG, soc_initial_kwh=limits["soc_min_kwh"])
    assert plan.soc_initial_kwh == limits["soc_min_kwh"]
    assert plan.schedule["soc_kwh"].iloc[-1] >= limits["soc_min_kwh"] - TOLERANCE


def test_infeasible_problem_raises_optimization_error():
    # Senza batteria né fotovoltaico, un carico oltre la potenza di rete non si può coprire.
    config = with_settings(battery={"capacity_kwh": 0.0})
    load = series(config["grid"]["max_power_kw"] + 10.0)
    with pytest.raises(OptimizationError):
        optimize_day(load, series(0.0), series(0.05), series(0.3), config)


def test_invalid_inputs_are_rejected():
    day = random_day(0)
    with pytest.raises(ValueError):
        optimize_day(day["load"].iloc[:-1], day["pv"], day["price"], day["carbon"], CONFIG)
    missing = day["pv"].copy()
    missing.iloc[3] = np.nan
    with pytest.raises(ValueError):
        optimize_day(day["load"], missing, day["price"], day["carbon"], CONFIG)
    with pytest.raises(ValueError):
        optimize_day(-day["load"], day["pv"], day["price"], day["carbon"], CONFIG)
    with pytest.raises(ValueError):
        optimize_day(**day, config=CONFIG, soc_initial_kwh=0.0)
