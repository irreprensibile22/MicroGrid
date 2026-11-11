"""Modello MILP per il piano orario di batteria e rete.

Uso:

    python -m microgrid.optimize    # risolve il giorno di esempio e ne disegna il piano

La formulazione è descritta in `docs/formulation.md`: i commenti qui sotto
riportano fra parentesi il numero che ogni vincolo ha in quel documento.
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
# Sotto questa soglia un valore restituito dal solver è considerato zero.
ZERO_TOLERANCE = 1e-7
# Colonna del piano -> nome della variabile nel modello.
SCHEDULE_VARIABLES = {
    "charge_kw": "p_ch",
    "discharge_kw": "p_dis",
    "grid_import_kw": "g_imp",
    "grid_export_kw": "g_exp",
    "curtailment_kw": "curt",
    "soc_kwh": "soc",
}


class OptimizationError(RuntimeError):
    """Il solver non ha trovato una soluzione ottima."""


@dataclass(frozen=True)
class Plan:
    """Piano orario di un giorno.

    `schedule` ha una riga per ogni ora, con lo stesso indice delle serie in
    ingresso. Le potenze [kW] sono medie sull'ora; `soc_kwh` è l'energia in
    batteria alla fine dell'ora. `cost_eur` ed `emissions_kg` sono quelli
    previsti dal piano, non quelli reali.
    """

    schedule: pd.DataFrame
    objective: float
    cost_eur: float
    emissions_kg: float
    soc_initial_kwh: float


def battery_limits(config: dict[str, Any]) -> dict[str, float]:
    """Limiti della batteria in unità fisiche (kWh e kW)."""
    battery = config["battery"]
    capacity = float(battery["capacity_kwh"])
    return {
        "soc_min_kwh": battery["soc_min_frac"] * capacity,
        "soc_max_kwh": battery["soc_max_frac"] * capacity,
        "soc_initial_kwh": battery["soc_initial_frac"] * capacity,
        # Una batteria senza capacità non può né caricare né scaricare.
        "max_power_kw": float(battery["max_power_kw"]) if capacity > 0 else 0.0,
    }


def exchange_prices(price: pd.Series, config: dict[str, Any]) -> tuple[pd.Series, pd.Series]:
    """Prezzi di acquisto e di vendita [EUR/kWh] a partire dal prezzo day-ahead."""
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
    """Costruisce il modello Pyomo di un giorno (potenze in kW, energie in kWh)."""
    battery, limits = config["battery"], battery_limits(config)
    step = float(config["time"]["step_hours"])
    grid_max = float(config["grid"]["max_power_kw"])
    power_max = limits["max_power_kw"]
    eta_charge, eta_discharge = battery["eta_charge"], battery["eta_discharge"]
    hours = len(load)

    model = pyo.ConcreteModel()
    model.T = pyo.RangeSet(1, hours)
    model.p_ch = pyo.Var(model.T, bounds=(0, power_max))
    model.p_dis = pyo.Var(model.T, bounds=(0, power_max))
    model.g_imp = pyo.Var(model.T, bounds=(0, grid_max))
    model.g_exp = pyo.Var(model.T, bounds=(0, grid_max))
    # (6) Non si può tagliare più di quanto si produce.
    model.curt = pyo.Var(model.T, bounds=lambda _, t: (0, float(pv[t - 1])))
    # (3) Limiti dello stato di carica.
    model.soc = pyo.Var(model.T, bounds=(limits["soc_min_kwh"], limits["soc_max_kwh"]))
    model.u = pyo.Var(model.T, domain=pyo.Binary)
    model.v = pyo.Var(model.T, domain=pyo.Binary)

    # (1) Bilancio di potenza.
    model.balance = pyo.Constraint(
        model.T,
        rule=lambda m, t: pv[t - 1] - m.curt[t] + m.p_dis[t] + m.g_imp[t]
        == load[t - 1] + m.p_ch[t] + m.g_exp[t],
    )

    # (2) Dinamica dello stato di carica.
    def soc_dynamics(m: pyo.ConcreteModel, t: int) -> Any:
        previous = soc_initial_kwh if t == 1 else m.soc[t - 1]
        return m.soc[t] == previous + eta_charge * m.p_ch[t] * step - m.p_dis[t] * step / eta_discharge

    model.soc_dynamics = pyo.Constraint(model.T, rule=soc_dynamics)

    # (4) Carica e scarica non simultanee.
    model.charge_only = pyo.Constraint(model.T, rule=lambda m, t: m.p_ch[t] <= power_max * m.u[t])
    model.discharge_only = pyo.Constraint(model.T, rule=lambda m, t: m.p_dis[t] <= power_max * (1 - m.u[t]))
    # (5) Acquisto e vendita non simultanei.
    model.import_only = pyo.Constraint(model.T, rule=lambda m, t: m.g_imp[t] <= grid_max * m.v[t])
    model.export_only = pyo.Constraint(model.T, rule=lambda m, t: m.g_exp[t] <= grid_max * (1 - m.v[t]))
    # (7) Fine giornata.
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
        raise ValueError("Le serie in ingresso devono avere la stessa lunghezza, non nulla.")
    for name, values in series.items():
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"La serie `{name}` contiene valori mancanti o non finiti.")
    for name in ("load", "pv", "carbon"):
        if (series[name] < 0).any():
            raise ValueError(f"La serie `{name}` contiene valori negativi.")
    if not limits["soc_min_kwh"] <= soc_initial_kwh <= limits["soc_max_kwh"]:
        raise ValueError(
            f"Stato di carica iniziale {soc_initial_kwh} kWh fuori dai limiti "
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
    """Calcola il piano ottimo di batteria e rete per le ore indicate.

    Parametri: carico e fotovoltaico previsti [kW], prezzo day-ahead
    [EUR/kWh] e intensità di carbonio [kgCO2/kWh], con lo stesso indice
    (23, 24 o 25 ore). `soc_initial_kwh` e `carbon_weight` (λ, EUR/kgCO2)
    hanno come default i valori della configurazione.

    Solleva `ValueError` se i dati in ingresso non sono validi e
    `OptimizationError` se il problema non ha soluzione ottima.
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
        raise OptimizationError(f"Solver `{settings['solver']}` non disponibile.")
    solver.config.load_solutions = False
    solver.config.rel_gap = settings["rel_gap"]
    solver.config.time_limit = settings["time_limit_s"]
    results = solver.solve(model, raise_exception_on_nonoptimal_result=False)
    if results.solution_status != SolutionStatus.optimal:
        raise OptimizationError(
            f"Nessuna soluzione ottima: {results.termination_condition.name} "
            f"(giorno che inizia a {load.index[0]})."
        )
    results.solution_loader.load_vars()

    schedule = pd.DataFrame({"load_kw": load.to_numpy(dtype=float), "pv_kw": pv.to_numpy(dtype=float)}, index=load.index)
    for column, variable in SCHEDULE_VARIABLES.items():
        values = np.array([pyo.value(getattr(model, variable)[t]) for t in model.T])
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
    # Import locali: servono solo per l'esempio, non per l'ottimizzatore.
    from microgrid.dataset import load_dataset, local_days
    from microgrid.forecast import load_forecasts
    from microgrid.plotting import apply_style, plot_plan

    config = load_config()
    day = pd.Timestamp(config["optimize"]["example_day"])
    dataset, forecasts = load_dataset(config), load_forecasts(config)
    hours = forecasts.index[local_days(forecasts.index, config) == day]
    if hours.empty:
        raise SystemExit(f"Nessuna previsione per il giorno {day.date()}.")
    plan = optimize_day(
        forecasts.loc[hours, "load_forecast"],
        forecasts.loc[hours, "pv_forecast"],
        dataset.loc[hours, "price_eur_kwh"],
        dataset.loc[hours, "carbon_kg_kwh"],
        config,
    )
    apply_style()
    figure = plot_plan(plan, config, title=f"Piano del {day.date()} (su carico e fotovoltaico previsti)")
    results_dir = resolve_path(config["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(results_dir / EXAMPLE_FILE, dpi=150, bbox_inches="tight")
    print(plan.schedule.round(2).to_string())
    print(f"costo previsto {plan.cost_eur:.2f} EUR, emissioni previste {plan.emissions_kg:.1f} kgCO2")


if __name__ == "__main__":
    main()
