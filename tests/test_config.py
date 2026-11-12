"""Consistency checks on the default configuration file."""

from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_battery_parameters_are_consistent():
    battery = load_config()["battery"]
    assert battery["capacity_kwh"] > 0
    assert battery["max_power_kw"] > 0
    assert 0 < battery["eta_charge"] <= 1
    assert 0 < battery["eta_discharge"] <= 1
    assert 0 <= battery["soc_min_frac"] <= battery["soc_initial_frac"] <= battery["soc_max_frac"] <= 1


def test_time_step_is_one_hour():
    # The hourly dataset and the 23/24/25-hour days both assume a one-hour step.
    assert load_config()["time"]["step_hours"] == 1
