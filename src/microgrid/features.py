"""Costruzione delle feature per i modelli di previsione.

La previsione delle 24 ore del giorno D+1 è emessa alla mezzanotte locale
fra D e D+1. Ogni feature usa quindi solo:

- informazioni di calendario del giorno D+1;
- meteo previsto per il giorno D+1 (qui: il meteo reale, vedi README);
- misure della grandezza da prevedere fino alla fine del giorno D.

Le misure dei giorni non validi non vengono mai usate come feature.
"""

from __future__ import annotations

from typing import Any

import holidays
import numpy as np
import pandas as pd

from microgrid.dataset import local_days, local_index

WEATHER_COLUMNS = ["temperature_c", "radiation_w_m2", "cloud_cover_pct"]


def day_starts(index: pd.DatetimeIndex, config: dict[str, Any]) -> pd.DatetimeIndex:
    """Per ogni ora, l'istante (UTC) della mezzanotte locale con cui inizia il suo giorno.

    È l'istante in cui viene emessa la previsione per quell'ora.
    """
    midnight = local_index(index, config).normalize()
    return midnight.tz_convert("UTC").rename("issue_time")


def calendar_features(index: pd.DatetimeIndex, config: dict[str, Any], weekday: bool = True) -> pd.DataFrame:
    """Feature di calendario, in ora locale, con codifica ciclica seno/coseno."""
    local = local_index(index, config)
    hour = local.hour + local.minute / 60
    day_of_year = local.day_of_year
    features = pd.DataFrame(
        {
            "hour": local.hour,
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "month": local.month,
            "year_sin": np.sin(2 * np.pi * day_of_year / 365.25),
            "year_cos": np.cos(2 * np.pi * day_of_year / 365.25),
        },
        index=index,
    )
    if weekday:
        site = config["site"]
        calendar = holidays.country_holidays(site["holidays_country"], subdiv=site.get("holidays_subdivision"))
        day_of_week = local.day_of_week
        features["day_of_week"] = day_of_week
        features["week_sin"] = np.sin(2 * np.pi * (day_of_week + hour / 24) / 7)
        features["week_cos"] = np.cos(2 * np.pi * (day_of_week + hour / 24) / 7)
        features["is_weekend"] = (day_of_week >= 5).astype(int)
        features["is_holiday"] = np.fromiter((day in calendar for day in local.date), dtype=int, count=len(local))
    return features


def lag_features(
    series: pd.Series, valid: pd.Series, lags_hours: list[int], issue_time: pd.DatetimeIndex
) -> pd.DataFrame:
    """Valori della serie di `lag` ore prima.

    Un valore ritardato è scartato (NaN) se proviene da un giorno non valido
    o se è stato misurato dall'istante di emissione in poi: succede solo
    nel giorno di 25 ore, la cui ultima ora dista 24 ore dalla prima.
    """
    usable = series.where(valid)
    features = {}
    for lag in lags_hours:
        source_time = series.index - pd.Timedelta(hours=lag)
        values = usable.reindex(source_time).to_numpy()
        known = source_time < issue_time
        features[f"lag_{lag}h"] = np.where(known, values, np.nan)
    return pd.DataFrame(features, index=series.index)


def history_features(series: pd.Series, valid: pd.Series, days: pd.Index, min_days: int) -> pd.DataFrame:
    """Medie della serie sul giorno precedente e sui 7 giorni precedenti.

    Sono uguali per tutte le ore dello stesso giorno. I giorni non validi
    non entrano nelle medie; la media settimanale richiede almeno
    `min_days` giorni validi.
    """
    usable = series.where(valid)
    daily = usable.groupby(days.to_numpy()).mean()
    daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq="D"))
    previous_day = daily.shift(1)
    previous_week = daily.shift(1).rolling(7, min_periods=min_days).mean()
    return pd.DataFrame(
        {
            "mean_previous_day": previous_day.reindex(days).to_numpy(),
            "mean_previous_7_days": previous_week.reindex(days).to_numpy(),
        },
        index=series.index,
    )


def weather_features(dataset: pd.DataFrame, days: pd.Index) -> pd.DataFrame:
    """Meteo dell'ora da prevedere e temperatura media del suo giorno."""
    features = dataset[WEATHER_COLUMNS].copy()
    features["temperature_day_mean"] = dataset["temperature_c"].groupby(days.to_numpy()).transform("mean")
    return features


def build_features(dataset: pd.DataFrame, config: dict[str, Any], target: str) -> pd.DataFrame:
    """Matrice delle feature per prevedere `target` (`load_kw` o `pv_kw`).

    Ha una riga per ogni ora del dataset. Le feature mancanti restano NaN:
    LightGBM le gestisce direttamente.
    """
    settings = config["forecast"]
    index = dataset.index
    days = local_days(index, config)
    valid = dataset["valid_day"] if target == "load_kw" else pd.Series(True, index=index)
    return pd.concat(
        [
            # Il giorno della settimana non influenza il sole: serve solo per il carico.
            calendar_features(index, config, weekday=target == "load_kw"),
            weather_features(dataset, days),
            lag_features(dataset[target], valid, settings["lags_hours"], day_starts(index, config)),
            history_features(dataset[target], valid, days, settings["min_days_weekly_mean"]),
        ],
        axis=1,
    )
