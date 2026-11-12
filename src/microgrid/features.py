"""Feature construction for the forecasting models.

The forecast for the 24 hours of day D+1 is issued at local midnight between
D and D+1. So every feature may only use:

- calendar information about day D+1;
- the weather forecast for day D+1 (here: the observed weather, see README);
- measurements of the target up to the end of day D.

Measurements from invalid days are never used as features.
"""

from __future__ import annotations

from typing import Any

import holidays
import numpy as np
import pandas as pd

from microgrid.dataset import local_days, local_index

WEATHER_COLUMNS = ["temperature_c", "radiation_w_m2", "cloud_cover_pct"]


def day_starts(index: pd.DatetimeIndex, config: dict[str, Any]) -> pd.DatetimeIndex:
    """For each hour, the UTC instant of the local midnight that starts its day.

    That is when the forecast for that hour is issued.
    """
    midnight = local_index(index, config).normalize()
    return midnight.tz_convert("UTC").rename("issue_time")


def calendar_features(index: pd.DatetimeIndex, config: dict[str, Any], weekday: bool = True) -> pd.DataFrame:
    """Calendar features in local time, with cyclic sine/cosine encoding."""
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
    """Values of the series `lag` hours earlier.

    A lagged value is dropped (NaN) if it comes from an invalid day, or if it
    was measured at or after the issue time. The latter only happens on the
    25-hour day, whose last hour is 24 hours after its first.
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
    """Mean of the series over the previous day and over the previous 7 days.

    Same value for every hour of a day. Invalid days are left out of the
    means; the weekly mean needs at least `min_days` valid days.
    """
    usable = series.where(valid)
    daily = usable.groupby(days.to_numpy()).mean()
    daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq="D"))
    # shift(1) so that a day never sees its own mean.
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
    """Weather of the hour to forecast, plus the mean temperature of its day."""
    features = dataset[WEATHER_COLUMNS].copy()
    features["temperature_day_mean"] = dataset["temperature_c"].groupby(days.to_numpy()).transform("mean")
    return features


def build_features(dataset: pd.DataFrame, config: dict[str, Any], target: str) -> pd.DataFrame:
    """Feature matrix for forecasting `target` (`load_kw` or `pv_kw`).

    One row per hour of the dataset. Missing features stay NaN: LightGBM
    handles them natively.
    """
    settings = config["forecast"]
    index = dataset.index
    days = local_days(index, config)
    valid = dataset["valid_day"] if target == "load_kw" else pd.Series(True, index=index)
    return pd.concat(
        [
            # The sun does not care what day of the week it is: weekday features are for load only.
            calendar_features(index, config, weekday=target == "load_kw"),
            weather_features(dataset, days),
            lag_features(dataset[target], valid, settings["lags_hours"], day_starts(index, config)),
            history_features(dataset[target], valid, days, settings["min_days_weekly_mean"]),
        ],
        axis=1,
    )
