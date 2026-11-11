"""Pulizia delle serie grezze e costruzione del dataset orario unico.

Uso:

    python -m microgrid.dataset

Legge i file di `data/raw/` e scrive `data/processed/dataset.parquet`, con
una riga per ogni ora del periodo di studio. L'indice è in UTC e indica
l'inizio dell'ora; per l'ora locale usare `local_index`.

Colonne:

- `load_kw`            carico della microgrid (riscalato) [kW]
- `pv_kw`              produzione fotovoltaica per la taglia configurata [kW]
- `temperature_c`      temperatura dell'aria all'inizio dell'ora [°C]
- `radiation_w_m2`     irraggiamento globale orizzontale medio nell'ora [W/m²]
- `cloud_cover_pct`    copertura nuvolosa all'inizio dell'ora [%]
- `price_eur_kwh`      prezzo day-ahead [EUR/kWh]
- `carbon_kg_kwh`      intensità di carbonio della rete [kgCO2eq/kWh]
- `valid_day`          False se il giorno locale contiene dati non attendibili
- `split`              `train`, `validation` o `test`
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from microgrid import data
from microgrid.config import load_config, resolve_path

DATASET_FILE = "dataset.parquet"
SUMMARY_FILE = "dataset_summary.json"
HOUR = pd.Timedelta(hours=1)
VALUE_COLUMNS = [
    "load_kw",
    "pv_kw",
    "temperature_c",
    "radiation_w_m2",
    "cloud_cover_pct",
    "price_eur_kwh",
    "carbon_kg_kwh",
]


# --- Indice temporale -------------------------------------------------------


def study_index(config: dict[str, Any]) -> pd.DatetimeIndex:
    """Indice orario in UTC che copre i giorni locali del periodo di studio.

    I giorni sono definiti nel fuso orario del sito: quelli del cambio
    dell'ora hanno quindi 23 o 25 ore.
    """
    timezone = config["site"]["timezone"]
    start = pd.Timestamp(config["data"]["start"], tz=timezone)
    end = pd.Timestamp(config["data"]["end"], tz=timezone) + pd.Timedelta(days=1)
    index = pd.date_range(start.tz_convert("UTC"), end.tz_convert("UTC"), freq="h", inclusive="left")
    return index.rename("utc_timestamp")


def local_index(index: pd.DatetimeIndex, config: dict[str, Any]) -> pd.DatetimeIndex:
    """Converte un indice UTC nel fuso orario del sito."""
    return index.tz_convert(config["site"]["timezone"])


def local_days(index: pd.DatetimeIndex, config: dict[str, Any]) -> pd.Index:
    """Giorno locale (mezzanotte, senza fuso) a cui appartiene ogni ora."""
    return pd.Index(local_index(index, config).tz_localize(None).normalize(), name="local_day")


# --- Operazioni elementari (funzioni pure) -----------------------------------


def run_lengths(mask: pd.Series) -> pd.Series:
    """Per ogni elemento True, lunghezza della sequenza di True consecutivi a cui appartiene."""
    mask = mask.astype(bool)
    group = (~mask).cumsum()
    return mask.groupby(group).transform("sum").where(mask, 0).astype(int)


def long_runs(mask: pd.Series, max_length: int) -> pd.Series:
    """True dove `mask` è True per più di `max_length` elementi consecutivi."""
    return run_lengths(mask) > max_length


def fill_short_gaps(series: pd.Series, max_gap: int) -> pd.Series:
    """Interpola linearmente i buchi lunghi al più `max_gap`; lascia NaN gli altri."""
    missing = series.isna()
    filled = series.interpolate(method="linear", limit_area="inside")
    return filled.where(~missing | (run_lengths(missing) <= max_gap))


def meter_to_power(readings: pd.Series) -> pd.Series:
    """Converte letture cumulative orarie di contatore [kWh] in potenza media [kW].

    La differenza fra la lettura dell'istante `t` e quella di `t − 1 h` è
    l'energia dell'ora che finisce in `t`: viene attribuita all'inizio di
    quell'ora, cioè a `t − 1 h`.
    """
    power = readings.diff()
    power.index = power.index - HOUR
    return power.iloc[1:]


def provider_interpolated(flags: pd.Series, column: str) -> pd.Series:
    """True dove il fornitore dichiara di avere interpolato la colonna `column`.

    `flags` è la colonna `interpolated` di Open Power System Data: elenca,
    separati da `|`, i nomi delle colonne interpolate in quell'istante.
    """
    pattern = rf"(?:^|\|)\s*{re.escape(column)}\s*(?:\||$)"
    return flags.fillna("").str.contains(pattern, regex=True)


def assign_split(valid_day: pd.Series, days: pd.Index, train: float, validation: float) -> pd.Series:
    """Divide i giorni in ordine temporale in train, validation e test.

    Le frazioni si applicano ai giorni interi, così nessun giorno è diviso
    fra due insiemi.
    """
    unique_days = pd.Index(days.unique()).sort_values()
    n_train = int(len(unique_days) * train)
    n_validation = int(len(unique_days) * validation)
    labels = pd.Series("test", index=unique_days)
    labels.iloc[:n_train] = "train"
    labels.iloc[n_train : n_train + n_validation] = "validation"
    return pd.Series(labels.reindex(days).to_numpy(), index=valid_day.index, name="split")


# --- Costruzione delle singole serie ----------------------------------------


def build_load(raw: pd.DataFrame, config: dict[str, Any], index: pd.DatetimeIndex) -> pd.DataFrame:
    """Carico aggregato delle utenze scelte, non riscalato.

    Restituisce `load_kw` e `load_gap`: quest'ultima è True nelle ore che
    fanno parte di un buco più lungo di `cleaning.max_gap_hours`, mancante o
    già interpolato dal fornitore, in almeno una delle colonne usate.
    """
    max_gap = config["cleaning"]["max_gap_hours"]
    total = pd.Series(0.0, index=index)
    gap = pd.Series(False, index=index)
    for household in config["load"]["households"]:
        consumption = pd.Series(0.0, index=index)
        for role, column in household.items():
            power = meter_to_power(raw[column]).reindex(index)
            sign = -1.0 if role == "export" else 1.0
            consumption = consumption + sign * power
            interpolated = provider_interpolated(raw["interpolated"], column)
            gap |= smoothed_hours(interpolated, max_gap).reindex(index, fill_value=False)
            gap |= long_runs(power.isna(), max_gap)
        # Un consumo negativo è fisicamente impossibile: è un errore di misura.
        consumption = consumption.where(consumption >= 0)
        total = total + consumption
    load = fill_short_gaps(total, max_gap)
    gap |= load.isna()
    return pd.DataFrame({"load_kw": load, "load_gap": gap})


def smoothed_hours(interpolated: pd.Series, max_gap: int) -> pd.Series:
    """Ore la cui energia è stata spalmata su un buco più lungo di `max_gap` ore.

    `interpolated` segnala le letture di contatore interpolate. Una sequenza
    di `n` letture interpolate consecutive ripartisce in parti uguali
    l'energia di `n + 1` ore: il totale è corretto, l'andamento no. Qui si
    segnalano le ore dei buchi con `n + 1 > max_gap`.

    Il risultato è riferito all'inizio dell'ora: l'ora che inizia in `t` usa
    le letture di `t` e di `t + 1 h`.
    """
    long_reading = run_lengths(interpolated) >= max_gap
    at_end = long_reading.shift(-1, fill_value=False)
    return (long_reading | at_end).iloc[:-1]


def build_pv(raw: pd.DataFrame, config: dict[str, Any], index: pd.DatetimeIndex) -> pd.Series:
    """Produzione fotovoltaica [kW] per la taglia configurata.

    Corregge i valori anomali: produzione negativa o produzione con
    irraggiamento nullo (di notte) vengono portate a zero.
    """
    pv = raw["pv_kw_per_kwp"].reindex(index) * config["pv"]["peak_kw"]
    dark = raw["irradiance_plane_w_m2"].reindex(index) <= 0
    return pv.clip(lower=0).mask(dark & pv.notna(), 0.0).rename("pv_kw")


def build_weather(raw: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Meteo orario con tutte le grandezze riferite all'inizio dell'ora.

    Open-Meteo fornisce la radiazione come media dell'ora precedente
    l'istante indicato: viene spostata indietro di un'ora per riferirla
    all'inizio dell'intervallo. Temperatura e nuvolosità sono istantanee.
    """
    radiation = raw["shortwave_radiation"].copy()
    radiation.index = radiation.index - HOUR
    return pd.DataFrame(
        {
            "temperature_c": raw["temperature_2m"].reindex(index),
            "radiation_w_m2": radiation.reindex(index).clip(lower=0),
            "cloud_cover_pct": raw["cloud_cover"].reindex(index),
        }
    )


def build_price(raw: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """Prezzo day-ahead [EUR/kWh]. I prezzi negativi sono reali e vengono mantenuti."""
    return (raw["price_eur_mwh"].reindex(index) / 1000.0).rename("price_eur_kwh")


def build_carbon(raw: pd.DataFrame, config: dict[str, Any], index: pd.DatetimeIndex) -> pd.Series:
    """Intensità di carbonio oraria [kgCO2eq/kWh] dal mix di produzione ogni 15 minuti."""
    hourly_generation = raw[list(config["emission_factors"])].resample("h").mean()
    intensity = data.carbon_intensity(hourly_generation, config["emission_factors"])
    return (intensity.reindex(index) / 1000.0).rename("carbon_kg_kwh")


# --- Dataset completo -------------------------------------------------------


def _read_raw(raw_dir: Path, filename: str) -> pd.DataFrame:
    path = raw_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Manca {path}: esegui prima `python -m microgrid.data`.")
    return pd.read_csv(path, index_col=0, parse_dates=True, low_memory=False)


def build_dataset(config: dict[str, Any] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Costruisce il dataset pulito e un riepilogo delle operazioni di pulizia."""
    config = config or load_config()
    raw_dir = resolve_path(config["paths"]["raw"])
    max_gap = config["cleaning"]["max_gap_hours"]
    index = study_index(config)

    load = build_load(_read_raw(raw_dir, data.LOAD_FILE), config, index)
    frame = pd.concat(
        [
            load["load_kw"],
            build_pv(_read_raw(raw_dir, data.PV_FILE), config, index),
            build_weather(_read_raw(raw_dir, data.WEATHER_FILE), index),
            build_price(_read_raw(raw_dir, data.PRICE_FILE), index),
            build_carbon(_read_raw(raw_dir, data.GENERATION_FILE), config, index),
        ],
        axis=1,
    )

    missing_before = frame.isna().sum()
    for column in VALUE_COLUMNS:
        frame[column] = fill_short_gaps(frame[column], max_gap)

    # Un giorno locale è valido se non ha buchi lunghi nel carico e nessun
    # valore mancante dopo l'interpolazione dei buchi brevi.
    days = local_days(index, config)
    bad_hour = load["load_gap"] | frame[VALUE_COLUMNS].isna().any(axis=1)
    frame["valid_day"] = ~bad_hour.groupby(days.to_numpy()).transform("any")

    # Riscalatura del carico al picco desiderato, calcolato sui giorni validi.
    real_load = frame["load_kw"].where(frame["valid_day"])
    target_peak = config["load"]["scale_to_peak_kw"]
    scale = 1.0 if target_peak is None else target_peak / real_load.max()
    frame["load_kw"] = frame["load_kw"] * scale

    frame["split"] = assign_split(
        frame["valid_day"], days, config["split"]["train"], config["split"]["validation"]
    )

    day_table = pd.DataFrame({"valid": frame["valid_day"].to_numpy(), "split": frame["split"].to_numpy()}, index=days)
    day_table = day_table[~day_table.index.duplicated()]
    summary = {
        "start_utc": index[0].isoformat(),
        "end_utc": index[-1].isoformat(),
        "hours": len(index),
        "days": len(day_table),
        "valid_days": int(day_table["valid"].sum()),
        "missing_before_cleaning": {k: int(v) for k, v in missing_before.items()},
        "load_real_mean_kw": float(real_load.mean()),
        "load_real_peak_kw": float(real_load.max()),
        "load_scale_factor": float(scale),
        "split_days": {
            name: {
                "days": int((day_table["split"] == name).sum()),
                "valid_days": int(((day_table["split"] == name) & day_table["valid"]).sum()),
                "first_day": str(day_table.index[day_table["split"] == name].min().date()),
                "last_day": str(day_table.index[day_table["split"] == name].max().date()),
            }
            for name in ("train", "validation", "test")
        },
    }
    return frame, summary


def load_dataset(config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Legge il dataset pulito da `data/processed/`."""
    config = config or load_config()
    path = resolve_path(config["paths"]["processed"]) / DATASET_FILE
    if not path.exists():
        raise FileNotFoundError(f"Manca {path}: esegui prima `python -m microgrid.dataset`.")
    return pd.read_parquet(path)


def main() -> None:
    config = load_config()
    frame, summary = build_dataset(config)
    processed_dir = resolve_path(config["paths"]["processed"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(processed_dir / DATASET_FILE)
    (processed_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
