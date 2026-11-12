"""Tests of the raw-data parsing functions (no network access)."""

import pandas as pd
import pytest

from microgrid.config import load_config
from microgrid.data import (
    carbon_intensity,
    parse_generation,
    parse_openmeteo,
    parse_price,
    parse_pvgis,
)


def test_parse_pvgis_normalises_power_and_floors_timestamps():
    payload = {
        "inputs": {"pv_module": {"peak_power": 2.0}},
        "outputs": {
            "hourly": [
                {"time": "20160101:1110", "P": 500.0, "G(i)": 300.0},
                {"time": "20160101:1210", "P": 1000.0, "G(i)": 600.0},
            ]
        },
    }
    out = parse_pvgis(payload)
    assert list(out.index) == [
        pd.Timestamp("2016-01-01 11:00", tz="UTC"),
        pd.Timestamp("2016-01-01 12:00", tz="UTC"),
    ]
    # 500 W from 2 kWp is 0.25 kW per kWp.
    assert out["pv_kw_per_kwp"].tolist() == [0.25, 0.5]


def test_parse_openmeteo_uses_utc_index():
    payload = {"hourly": {"time": ["2016-05-14T00:00", "2016-05-14T01:00"], "temperature_2m": [9.5, 9.1]}}
    out = parse_openmeteo(payload)
    assert out.index[0] == pd.Timestamp("2016-05-14 00:00", tz="UTC")
    assert out["temperature_2m"].tolist() == [9.5, 9.1]


def test_parse_price_converts_unix_seconds():
    payload = {"unix_seconds": [1463176800, 1463180400], "price": [20.0, -5.0], "unit": "EUR / MWh"}
    out = parse_price(payload)
    assert out.index[0] == pd.Timestamp("2016-05-13 22:00", tz="UTC")
    assert out["price_eur_mwh"].tolist() == [20.0, -5.0]


def test_parse_price_rejects_unexpected_unit():
    with pytest.raises(ValueError):
        parse_price({"unix_seconds": [0], "price": [1.0], "unit": "EUR / kWh"})


def test_carbon_intensity_is_power_weighted_mean():
    payload = {
        "unix_seconds": [0, 3600, 7200],
        "production_types": [
            {"name": "Fossil gas", "data": [100.0, 0.0, 50.0]},
            {"name": "Wind onshore", "data": [100.0, 200.0, -10.0]},
            {"name": "Load", "data": [500.0, 500.0, 500.0]},
        ],
    }
    generation = parse_generation(payload)
    intensity = carbon_intensity(generation, {"Fossil gas": 500.0, "Wind onshore": 10.0})
    # Hour 0: half gas, half wind. Hour 1: wind only.
    # Hour 2: negative generation counts as zero, so only gas is left.
    assert intensity.tolist() == pytest.approx([255.0, 10.0, 500.0])


def test_carbon_intensity_requires_all_configured_sources():
    generation = pd.DataFrame({"Fossil gas": [1.0]})
    with pytest.raises(KeyError):
        carbon_intensity(generation, {"Fossil gas": 490.0, "Nuclear": 12.0})


def test_configured_households_have_an_import_column():
    households = load_config()["load"]["households"]
    assert households
    for household in households:
        assert "import" in household
        # Consumption behind the meter can only be rebuilt with both columns.
        assert ("pv" in household) == ("export" in household)
