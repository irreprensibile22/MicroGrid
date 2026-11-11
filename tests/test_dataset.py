"""Test delle operazioni di pulizia (senza file di dati)."""

import numpy as np
import pandas as pd
import pytest

from microgrid.config import load_config
from microgrid.dataset import (
    assign_split,
    build_load,
    build_pv,
    build_weather,
    fill_short_gaps,
    local_days,
    long_runs,
    meter_to_power,
    provider_interpolated,
    run_lengths,
    smoothed_hours,
    study_index,
)


def hours(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq="h", tz="UTC")


def test_run_lengths_counts_consecutive_true_values():
    mask = pd.Series([True, True, False, True, False, False, True, True, True])
    assert run_lengths(mask).tolist() == [2, 2, 0, 1, 0, 0, 3, 3, 3]
    assert long_runs(mask, 2).tolist() == [False] * 6 + [True] * 3


def test_fill_short_gaps_interpolates_only_short_gaps():
    series = pd.Series([1.0, np.nan, 3.0, np.nan, np.nan, np.nan, 7.0])
    filled = fill_short_gaps(series, max_gap=2)
    assert filled.iloc[1] == pytest.approx(2.0)
    assert filled.iloc[3:6].isna().all()
    assert filled.iloc[[0, 2, 6]].tolist() == [1.0, 3.0, 7.0]


def test_meter_to_power_assigns_energy_to_start_of_hour():
    readings = pd.Series([10.0, 12.0, 12.5], index=hours("2016-01-01 00:00", 3))
    power = meter_to_power(readings)
    # 2 kWh fra le 00:00 e le 01:00 sono 2 kW nell'ora che inizia alle 00:00.
    assert power.index.tolist() == hours("2016-01-01 00:00", 2).tolist()
    assert power.tolist() == [2.0, 0.5]


def test_provider_interpolated_matches_whole_column_names():
    flags = pd.Series(["house1_pv | house1_grid_import", "house1_grid_import_extra", None])
    assert provider_interpolated(flags, "house1_grid_import").tolist() == [True, False, False]


def test_smoothed_hours_flags_only_long_interpolated_runs():
    index = hours("2016-01-01 00:00", 10)
    # Due letture interpolate consecutive (ore 3 e 4) spalmano l'energia su 3 ore: 2, 3, 4.
    interpolated = pd.Series(False, index=index)
    interpolated.iloc[[3, 4]] = True
    assert smoothed_hours(interpolated, max_gap=3).sum() == 0
    flagged = smoothed_hours(interpolated, max_gap=2)
    assert flagged[flagged].index.tolist() == index[[2, 3, 4]].tolist()


def test_study_index_follows_local_days_across_dst_changes():
    config = {"site": {"timezone": "Europe/Berlin"}, "data": {"start": "2016-03-26", "end": "2016-03-28"}}
    index = study_index(config)
    assert index[0] == pd.Timestamp("2016-03-25 23:00", tz="UTC")
    per_day = pd.Series(1, index=local_days(index, config)).groupby(level=0).sum()
    # Il 27 marzo 2016 inizia l'ora legale: il giorno ha 23 ore.
    assert per_day.tolist() == [24, 23, 24]


def test_assign_split_is_chronological_and_keeps_days_whole():
    index = hours("2016-01-01 00:00", 24 * 20)
    days = pd.Index(index.tz_localize(None).normalize())
    split = assign_split(pd.Series(True, index=index), days, train=0.70, validation=0.15)
    assert split.groupby(days.to_numpy()).nunique().max() == 1
    assert split.groupby(days.to_numpy()).first().tolist() == ["train"] * 14 + ["validation"] * 3 + ["test"] * 3


def test_build_load_reconstructs_consumption_behind_the_meter():
    index = hours("2016-01-01 00:00", 4)
    readings = hours("2016-01-01 00:00", 5)
    raw = pd.DataFrame(
        {
            "a_import": [0.0, 1.0, 2.0, 3.0, 4.0],
            "b_import": [0.0, 0.0, 0.0, 1.0, 2.0],
            "b_pv": [0.0, 0.0, 3.0, 5.0, 5.0],
            "b_export": [0.0, 0.0, 2.0, 3.0, 3.0],
            "interpolated": [None] * 5,
        },
        index=readings,
    )
    config = {
        "cleaning": {"max_gap_hours": 2},
        "load": {"households": [{"import": "a_import"}, {"import": "b_import", "pv": "b_pv", "export": "b_export"}]},
    }
    load = build_load(raw, config, index)
    # Utenza b: prelievo + produzione − immissione = [0, 1, 2, 1]; utenza a: 1 kW costante.
    assert load["load_kw"].tolist() == [1.0, 2.0, 3.0, 2.0]
    assert not load["load_gap"].any()


def test_build_pv_scales_to_peak_and_removes_night_production():
    index = hours("2016-06-01 00:00", 3)
    raw = pd.DataFrame(
        {"pv_kw_per_kwp": [0.02, 0.5, -0.1], "irradiance_plane_w_m2": [0.0, 600.0, 50.0]}, index=index
    )
    pv = build_pv(raw, {"pv": {"peak_kw": 100.0}}, index)
    assert pv.tolist() == [0.0, 50.0, 0.0]


def test_build_weather_moves_radiation_to_start_of_hour():
    raw_index = hours("2016-06-01 10:00", 3)
    raw = pd.DataFrame(
        {"temperature_2m": [15.0, 16.0, 17.0], "shortwave_radiation": [100.0, 200.0, 300.0], "cloud_cover": [0, 50, 100]},
        index=raw_index,
    )
    weather = build_weather(raw, raw_index[:2])
    # La radiazione indicata alle 11:00 è la media fra le 10:00 e le 11:00.
    assert weather["radiation_w_m2"].tolist() == [200.0, 300.0]
    assert weather["temperature_c"].tolist() == [15.0, 16.0]


def test_default_split_fractions_leave_room_for_test_set():
    split = load_config()["split"]
    assert 0 < split["train"] and 0 < split["validation"]
    assert split["train"] + split["validation"] < 1
