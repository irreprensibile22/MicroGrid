"""Simulazione a orizzonte mobile: un piano al giorno, eseguito sui dati reali.

Uso:

    python -m microgrid.simulate

Per ogni giorno del periodo simulato:

1. il MILP calcola il piano della batteria su carico e fotovoltaico previsti;
2. il piano è eseguito sui valori reali: la batteria segue il programma,
   la differenza fra previsto e reale è assorbita dalla rete;
3. l'energia rimasta in batteria è lo stato iniziale del giorno successivo.

Scrive il dettaglio orario in `data/processed/simulation.parquet` e la serie
giornaliera di costi ed emissioni in `results/simulation_daily.csv`.

Regole di correzione applicate in tempo reale, ora per ora (`apply_plan`):

- la batteria non esce dai limiti dello stato di carica: carica e scarica
  programmate sono ridotte a quanto è fisicamente possibile;
- se l'acquisto supererebbe la potenza di rete, si riduce la carica;
- il surplus si vende; si taglia il fotovoltaico se il prezzo di vendita è
  negativo o se la vendita supererebbe la potenza di rete.

Con la correzione in tempo reale (`correct=True`) si segue lo scambio con la
rete previsto dal piano, non il programma della batteria: l'errore di
previsione è assorbito dalla batteria, finché può. Due limiti evitano gli
scambi svantaggiosi:

- la batteria carica al più il surplus reale, più la carica dalla rete che
  il piano aveva deciso;
- la batteria scarica al più il deficit reale, più la scarica verso la rete
  che il piano aveva deciso.
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
    """Il carico reale non può essere coperto rispettando i limiti dell'impianto."""


def apply_plan(
    charge: pd.Series,
    discharge: pd.Series,
    actual: pd.DataFrame,
    config: dict[str, Any],
    soc_initial_kwh: float,
    schedule: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Esegue il programma della batteria sui valori reali di un periodo.

    `charge` e `discharge` sono le potenze programmate [kW]; `actual` contiene
    `load_kw`, `pv_kw`, `price_eur_kwh` e `carbon_kg_kwh` reali, con lo stesso
    indice. Restituisce, per ogni ora, i flussi realmente avvenuti, l'energia
    in batteria a fine ora, il costo [EUR] e le emissioni [kgCO2].

    Se è indicato il piano completo (`schedule`, come restituito da
    `optimize_day`), si applica la correzione in tempo reale descritta in
    testa al modulo.
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
            # Componenti del piano che scambiano con la rete di proposito.
            from_grid = charge_kw - min(charge_kw, max(planned_surplus, 0.0))
            to_grid = discharge_kw - min(discharge_kw, max(-planned_surplus, 0.0))
            # Potenza che la batteria dovrebbe cedere (+) o assorbire (−) per
            # scambiare con la rete quanto previsto dal piano.
            planned_exchange = float(planned["grid_import_kw"] - planned["grid_export_kw"])
            needed = (load - pv) - planned_exchange
            discharge_kw = min(max(needed, 0.0), max(load - pv, 0.0) + to_grid, power_max)
            charge_kw = min(max(-needed, 0.0), max(pv - load, 0.0) + from_grid, power_max)
        # La batteria segue il programma, entro ciò che lo stato di carica consente.
        charge_kw = min(charge_kw, max(limits["soc_max_kwh"] - soc, 0.0) / (eta_charge * step))
        discharge_kw = min(discharge_kw, max(soc - limits["soc_min_kwh"], 0.0) * eta_discharge / step)

        # La rete assorbe la differenza fra previsto e reale.
        net = load - pv + charge_kw - discharge_kw
        if net > grid_max:
            charge_kw = max(charge_kw - (net - grid_max), 0.0)
            net = load - pv + charge_kw - discharge_kw
            if net > grid_max + ZERO_TOLERANCE:
                raise SimulationError(
                    f"Alle {actual.index[hour]} servono {net:.1f} kW dalla rete, più del massimo di {grid_max:.1f} kW."
                )
        grid_import = max(net, 0.0)
        surplus = max(-net, 0.0)
        # Vendere a prezzo negativo costa: si taglia il fotovoltaico, fin dove basta.
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
    """Totali di un giorno a partire dai valori reali e dai flussi avvenuti."""
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
    """Simula le strategie senza previsione né ottimizzazione.

    Con `use_battery=True` è la regola ingenua: la batteria carica quando il
    fotovoltaico supera il carico e scarica quando non basta, entro i suoi
    limiti. Con `use_battery=False` la batteria non viene mai usata.
    Restituisce dettaglio orario e serie giornaliera, come `simulate`.
    """
    if inputs[INPUT_COLUMNS].isna().any().any():
        raise ValueError("I dati della simulazione contengono valori mancanti.")
    step = float(config["time"]["step_hours"])
    limits = battery_limits(config)
    power = limits["max_power_kw"] if use_battery else 0.0
    surplus = inputs["pv_kw"] - inputs["load_kw"]
    soc = limits["soc_initial_kwh"]
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
    """Simula giorno per giorno le ore di `inputs`, che devono essere consecutive.

    `inputs` contiene i valori reali (`load_kw`, `pv_kw`, `price_eur_kwh`,
    `carbon_kg_kwh`) e le colonne `load_column` e `pv_column` su cui il MILP
    pianifica. Passando `load_kw` e `pv_kw` anche come colonne di
    pianificazione si ottiene il caso di previsione perfetta.

    Con `correct=True` il programma è eseguito con la correzione in tempo
    reale descritta in testa al modulo.

    Restituisce il dettaglio orario e la serie giornaliera. Nel dettaglio
    le colonne `planned_*` sono il programma, le altre ciò che è avvenuto.
    """
    needed = [*INPUT_COLUMNS, load_column, pv_column]
    if inputs[needed].isna().any().any():
        raise ValueError("I dati della simulazione contengono valori mancanti.")
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
        # Lo stato reale di fine giornata è quello iniziale del giorno dopo.
        soc = float(np.clip(real["soc_kwh"].iloc[-1], *_soc_bounds(config)))
    return pd.concat(hourly), pd.DataFrame(daily).set_index("day")


def _soc_bounds(config: dict[str, Any]) -> tuple[float, float]:
    limits = battery_limits(config)
    return limits["soc_min_kwh"], limits["soc_max_kwh"]


def simulation_inputs(config: dict[str, Any], split: str) -> pd.DataFrame:
    """Valori reali e previsioni delle ore di `split` (`validation` o `test`)."""
    dataset, forecasts = load_dataset(config), load_forecasts(config)
    hours = forecasts.index[forecasts["split"] == split]
    return pd.concat(
        [dataset.loc[hours, INPUT_COLUMNS], forecasts.loc[hours, ["load_forecast", "pv_forecast"]]], axis=1
    )


def load_simulation(config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Legge il dettaglio orario salvato da `python -m microgrid.simulate`."""
    config = config or load_config()
    path = resolve_path(config["paths"]["processed"]) / HOURLY_FILE
    if not path.exists():
        raise FileNotFoundError(f"Manca {path}: esegui prima `python -m microgrid.simulate`.")
    return pd.read_parquet(path)


def main() -> None:
    """Simula il periodo configurato e salva i risultati (comando `python -m microgrid.simulate`)."""
    config = load_config()
    split = config["simulate"]["split"]
    hourly, daily = simulate(simulation_inputs(config, split), config)
    results_dir = resolve_path(config["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(resolve_path(config["paths"]["processed"]) / HOURLY_FILE)
    daily.to_csv(results_dir / DAILY_FILE, float_format="%.4f")
    print(daily.round(2).to_string())
    print(
        f"\n{split}: {len(daily)} giorni, costo {daily['cost_eur'].sum():.2f} EUR "
        f"(previsto {daily['planned_cost_eur'].sum():.2f}), emissioni {daily['emissions_kg'].sum():.1f} kgCO2"
    )


if __name__ == "__main__":
    main()
