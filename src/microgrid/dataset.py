"""Cleaning of the raw series and construction of the single hourly dataset.

Usage:

    python -m microgrid.dataset

Reads the files in `data/raw/` and writes `data/processed/dataset.parquet`,
one row per hour of the study period. The index is UTC and marks the start of
the hour; use `local_index` for local time.

Columns:

- `load_kw`            microgrid load (rescaled) [kW]
- `pv_kw`              PV output for the configured plant size [kW]
- `temperature_c`      air temperature at the start of the hour [°C]
- `radiation_w_m2`     mean global horizontal irradiance over the hour [W/m²]
- `cloud_cover_pct`    cloud cover at the start of the hour [%]
- `price_eur_kwh`      day-ahead price [EUR/kWh]
- `carbon_kg_kwh`      grid carbon intensity [kgCO2eq/kWh]
- `valid_day`          False if the local day contains unreliable data
- `split`              `train`, `validation` or `test`
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


# --- Time index --------------------------------------------------------------


def study_index(config: dict[str, Any]) -> pd.DatetimeIndex:
    """Hourly UTC index covering the local days of the study period.

    Days are defined in the site's time zone, so the days of the clock change
    have 23 or 25 hours.
    """
    timezone = config["site"]["timezone"]
    start = pd.Timestamp(config["data"]["start"], tz=timezone)
    end = pd.Timestamp(config["data"]["end"], tz=timezone) + pd.Timedelta(days=1)
    index = pd.date_range(start.tz_convert("UTC"), end.tz_convert("UTC"), freq="h", inclusive="left")
    return index.rename("utc_timestamp")


def local_index(index: pd.DatetimeIndex, config: dict[str, Any]) -> pd.DatetimeIndex:
    """Convert a UTC index to the site's time zone."""
    return index.tz_convert(config["site"]["timezone"])


def local_days(index: pd.DatetimeIndex, config: dict[str, Any]) -> pd.Index:
    """Local day (midnight, time zone dropped) each hour belongs to."""
    return pd.Index(local_index(index, config).tz_localize(None).normalize(), name="local_day")


# --- Building blocks (pure functions) ----------------------------------------


def run_lengths(mask: pd.Series) -> pd.Series:
    """For each True element, the length of the run of consecutive Trues it belongs to."""
    mask = mask.astype(bool)
    group = (~mask).cumsum()
    return mask.groupby(group).transform("sum").where(mask, 0).astype(int)


def long_runs(mask: pd.Series, max_length: int) -> pd.Series:
    """True where `mask` is True for more than `max_length` consecutive elements."""
    return run_lengths(mask) > max_length


def fill_short_gaps(series: pd.Series, max_gap: int) -> pd.Series:
    """Linearly interpolate gaps up to `max_gap` long; leave longer ones as NaN."""
    missing = series.isna()
    filled = series.interpolate(method="linear", limit_area="inside")
    return filled.where(~missing | (run_lengths(missing) <= max_gap))


def meter_to_power(readings: pd.Series) -> pd.Series:
    """Convert hourly cumulative meter readings [kWh] into mean power [kW].

    The difference between the reading at `t` and the one at `t − 1 h` is the
    energy of the hour ending at `t`. It is assigned to the start of that
    hour, i.e. to `t − 1 h`.
    """
    power = readings.diff()
    power.index = power.index - HOUR
    return power.iloc[1:]


def provider_interpolated(flags: pd.Series, column: str) -> pd.Series:
    """True where the data provider says it interpolated `column`.

    `flags` is the `interpolated` column of Open Power System Data: a
    `|`-separated list of the columns interpolated at that instant.
    """
    # Match whole names only: `x_grid_import` must not match `x_grid_import_extra`.
    pattern = rf"(?:^|\|)\s*{re.escape(column)}\s*(?:\||$)"
    return flags.fillna("").str.contains(pattern, regex=True)


def assign_split(valid_day: pd.Series, days: pd.Index, train: float, validation: float) -> pd.Series:
    """Split the days chronologically into train, validation and test.

    The fractions apply to whole days, so no day is shared between two sets.
    """
    unique_days = pd.Index(days.unique()).sort_values()
    n_train = int(len(unique_days) * train)
    n_validation = int(len(unique_days) * validation)
    labels = pd.Series("test", index=unique_days)
    labels.iloc[:n_train] = "train"
    labels.iloc[n_train : n_train + n_validation] = "validation"
    return pd.Series(labels.reindex(days).to_numpy(), index=valid_day.index, name="split")


# --- Individual series -------------------------------------------------------


def build_load(raw: pd.DataFrame, config: dict[str, Any], index: pd.DatetimeIndex) -> pd.DataFrame:
    """Aggregate load of the chosen households, not rescaled.

    Returns `load_kw` and `load_gap`. The latter is True for hours that are
    part of a gap longer than `cleaning.max_gap_hours` (missing, or already
    interpolated by the provider) in at least one of the columns used.
    """
    max_gap = config["cleaning"]["max_gap_hours"]
    total = pd.Series(0.0, index=index)
    gap = pd.Series(False, index=index)
    for household in config["load"]["households"]:
        # Consumption behind the meter: grid import + own PV − grid export.
        consumption = pd.Series(0.0, index=index)
        for role, column in household.items():
            power = meter_to_power(raw[column]).reindex(index)
            sign = -1.0 if role == "export" else 1.0
            consumption = consumption + sign * power
            interpolated = provider_interpolated(raw["interpolated"], column)
            gap |= smoothed_hours(interpolated, max_gap).reindex(index, fill_value=False)
            gap |= long_runs(power.isna(), max_gap)
        # Negative consumption is physically impossible: treat it as a metering error.
        consumption = consumption.where(consumption >= 0)
        total = total + consumption
    load = fill_short_gaps(total, max_gap)
    gap |= load.isna()
    return pd.DataFrame({"load_kw": load, "load_gap": gap})


def smoothed_hours(interpolated: pd.Series, max_gap: int) -> pd.Series:
    """Hours whose energy was spread over a gap longer than `max_gap` hours.

    `interpolated` flags interpolated meter readings. A run of `n` consecutive
    interpolated readings spreads the energy of `n + 1` hours evenly: the
    total is right, the hourly shape is not. This flags the hours of gaps
    with `n + 1 > max_gap`.

    The result refers to the start of the hour: the hour starting at `t` uses
    the readings at `t` and `t + 1 h`.
    """
    long_reading = run_lengths(interpolated) >= max_gap
    at_end = long_reading.shift(-1, fill_value=False)
    return (long_reading | at_end).iloc[:-1]


def build_pv(raw: pd.DataFrame, config: dict[str, Any], index: pd.DatetimeIndex) -> pd.Series:
    """PV output [kW] for the configured plant size.

    Negative output, and output with zero irradiance (at night), are set to zero.
    """
    pv = raw["pv_kw_per_kwp"].reindex(index) * config["pv"]["peak_kw"]
    dark = raw["irradiance_plane_w_m2"].reindex(index) <= 0
    return pv.clip(lower=0).mask(dark & pv.notna(), 0.0).rename("pv_kw")


def build_weather(raw: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Hourly weather with every quantity referred to the start of the hour.

    Open-Meteo reports radiation as the mean of the hour *before* the given
    instant, so it is shifted back by one hour. Temperature and cloud cover
    are instantaneous.
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
    """Day-ahead price [EUR/kWh]. Negative prices are real and are kept."""
    return (raw["price_eur_mwh"].reindex(index) / 1000.0).rename("price_eur_kwh")


def build_carbon(raw: pd.DataFrame, config: dict[str, Any], index: pd.DatetimeIndex) -> pd.Series:
    """Hourly carbon intensity [kgCO2eq/kWh] from the 15-minute generation mix."""
    hourly_generation = raw[list(config["emission_factors"])].resample("h").mean()
    intensity = data.carbon_intensity(hourly_generation, config["emission_factors"])
    return (intensity.reindex(index) / 1000.0).rename("carbon_kg_kwh")


# --- Full dataset ------------------------------------------------------------


def _read_raw(raw_dir: Path, filename: str) -> pd.DataFrame:
    path = raw_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}: run `python -m microgrid.data` first.")
    return pd.read_csv(path, index_col=0, parse_dates=True, low_memory=False)


def build_dataset(config: dict[str, Any] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the clean dataset and a summary of what the cleaning did."""
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

    # A local day is valid if the load has no long gaps and nothing is still
    # missing once the short gaps have been filled.
    days = local_days(index, config)
    bad_hour = load["load_gap"] | frame[VALUE_COLUMNS].isna().any(axis=1)
    frame["valid_day"] = ~bad_hour.groupby(days.to_numpy()).transform("any")

    # Rescale the load to the target peak, measured on valid days only.
    real_load = frame["load_kw"].where(frame["valid_day"])
    target_peak = config["load"]["scale_to_peak_kw"]
    scale = 1.0 if target_peak is None else target_peak / real_load.max()
    frame["load_kw"] = frame["load_kw"] * scale

    frame["split"] = assign_split(frame["valid_day"], days, config["split"]["train"], config["split"]["validation"])

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
    """Read the clean dataset from `data/processed/`."""
    config = config or load_config()
    path = resolve_path(config["paths"]["processed"]) / DATASET_FILE
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}: run `python -m microgrid.dataset` first.")
    return pd.read_parquet(path)


def main() -> None:
    """Build and save the clean dataset (`python -m microgrid.dataset`)."""
    config = load_config()
    frame, summary = build_dataset(config)
    processed_dir = resolve_path(config["paths"]["processed"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(processed_dir / DATASET_FILE)
    (processed_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
