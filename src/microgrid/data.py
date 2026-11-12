"""Download of the raw hourly series (load, PV, weather, prices, generation mix).

Usage:

    python -m microgrid.data            # download whatever is missing in data/raw/
    python -m microgrid.data --force    # download everything again

Each `download_*` function saves one file in `data/raw/` and returns its path.
All timestamps are UTC and mark the start of the interval. Cleaning and
merging happen later, in `microgrid.dataset`.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

from microgrid.config import load_config, resolve_path

USER_AGENT = "microgrid-ems"
# One extra day on each side: local time vs UTC, plus the lagged features of the first days.
MARGIN = pd.Timedelta(days=1)

LOAD_FILE = "opsd_household_60min.csv"
PV_FILE = "pvgis_hourly_per_kwp.csv"
WEATHER_FILE = "openmeteo_hourly.csv"
PRICE_FILE = "day_ahead_price.csv"
GENERATION_FILE = "generation_by_type.csv"


# --- Network ----------------------------------------------------------------


def _fetch(url: str, params: dict[str, Any] | None = None, retries: int = 6) -> bytes:
    """Download a URL, retrying with a growing delay on temporary errors."""
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    wait = 5.0
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            # A 4xx other than 429 means the request itself is wrong: retrying is pointless.
            if error.code < 500 and error.code != 429:
                raise
            last_error: Exception = error
            # On 429 the service tells us how long to wait.
            retry_after = error.headers.get("Retry-After", "")
            wait = float(retry_after) + 1 if retry_after.isdigit() else wait
        except (urllib.error.URLError, TimeoutError) as error:
            last_error = error
        if attempt < retries - 1:
            time.sleep(wait)
            wait = 5 * 2 ** (attempt + 1)
    raise RuntimeError(f"Download failed after {retries} attempts: {url}") from last_error


def _fetch_json(url: str, params: dict[str, Any] | None = None) -> Any:
    return json.loads(_fetch(url, params))


def _period(config: dict[str, Any]) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last date to download (study period plus margin)."""
    start = pd.Timestamp(config["data"]["start"]) - MARGIN
    end = pd.Timestamp(config["data"]["end"]) + MARGIN
    return start, end


# --- Response parsing (pure functions, testable offline) ---------------------


def parse_pvgis(payload: dict[str, Any]) -> pd.DataFrame:
    """Convert the PVGIS hourly response into power per installed kWp.

    PVGIS timestamps look like `YYYYMMDD:HHMM` in UTC, where the minutes are
    those of the satellite pass (e.g. `:10`). The value is assigned to the
    hour containing that instant. Power `P` is in W.
    """
    hourly = pd.DataFrame(payload["outputs"]["hourly"])
    index = pd.to_datetime(hourly["time"], format="%Y%m%d:%H%M", utc=True).dt.floor("h")
    peak_kw = float(payload["inputs"]["pv_module"]["peak_power"])
    out = pd.DataFrame(
        {
            "pv_kw_per_kwp": hourly["P"].to_numpy() / 1000.0 / peak_kw,
            "irradiance_plane_w_m2": hourly["G(i)"].to_numpy(),
        },
        index=pd.DatetimeIndex(index, name="utc_timestamp"),
    )
    return out


def parse_openmeteo(payload: dict[str, Any]) -> pd.DataFrame:
    """Convert the Open-Meteo hourly response (requested in UTC) into a table."""
    hourly = pd.DataFrame(payload["hourly"])
    index = pd.to_datetime(hourly.pop("time"), utc=True)
    hourly.index = pd.DatetimeIndex(index, name="utc_timestamp")
    return hourly


def parse_price(payload: dict[str, Any]) -> pd.DataFrame:
    """Convert the Energy-Charts `/price` response into prices [EUR/MWh]."""
    if payload.get("unit") != "EUR / MWh":
        raise ValueError(f"Unexpected price unit: {payload.get('unit')!r}")
    index = pd.to_datetime(payload["unix_seconds"], unit="s", utc=True)
    return pd.DataFrame(
        {"price_eur_mwh": payload["price"]},
        index=pd.DatetimeIndex(index, name="utc_timestamp"),
    )


def parse_generation(payload: dict[str, Any]) -> pd.DataFrame:
    """Convert the Energy-Charts `/public_power` response into power [MW] per source."""
    index = pd.to_datetime(payload["unix_seconds"], unit="s", utc=True)
    columns = {series["name"]: series["data"] for series in payload["production_types"]}
    return pd.DataFrame(columns, index=pd.DatetimeIndex(index, name="utc_timestamp"))


def carbon_intensity(generation: pd.DataFrame, factors: dict[str, float]) -> pd.Series:
    """Carbon intensity of generation [gCO2eq/kWh].

    Mean of the emission factors, weighted by the power produced by each
    source. Only the sources listed in `factors` count. Negative generation
    (e.g. pumped storage consuming) is treated as zero. Cross-border flows
    are ignored.
    """
    missing = sorted(set(factors) - set(generation.columns))
    if missing:
        raise KeyError(f"Sources without generation data: {missing}")
    power = generation[list(factors)].clip(lower=0)
    weights = pd.Series(factors, dtype=float)
    total = power.sum(axis=1, min_count=1)
    intensity = power.mul(weights, axis=1).sum(axis=1, min_count=1) / total
    return intensity.rename("carbon_intensity_g_kwh")


# --- Downloads ---------------------------------------------------------------


def download_load(config: dict[str, Any], raw_dir: Path) -> Path:
    """Download the hourly household measurements for Konstanz (Open Power System Data).

    The file is saved as is: cumulative meter readings [kWh] for every
    household and appliance.
    """
    path = raw_dir / LOAD_FILE
    path.write_bytes(_fetch(config["load"]["source_url"]))
    return path


def download_pv(config: dict[str, Any], raw_dir: Path) -> Path:
    """Download the PV output simulated by PVGIS, per installed kWp.

    Storing it per kWp lets us change the plant size without downloading again.
    """
    start, end = _period(config)
    site, pvgis = config["site"], config["pvgis"]
    payload = _fetch_json(
        pvgis["url"],
        {
            "lat": site["latitude"],
            "lon": site["longitude"],
            "pvcalculation": 1,
            "peakpower": 1,
            "loss": pvgis["loss_percent"],
            "angle": pvgis["tilt_deg"],
            "aspect": pvgis["azimuth_deg"],
            "startyear": start.year,
            "endyear": end.year,
            "outputformat": "json",
        },
    )
    path = raw_dir / PV_FILE
    parse_pvgis(payload).to_csv(path)
    return path


def download_weather(config: dict[str, Any], raw_dir: Path) -> Path:
    """Download historical hourly weather from Open-Meteo (reanalysis)."""
    start, end = _period(config)
    site, weather = config["site"], config["weather"]
    payload = _fetch_json(
        weather["url"],
        {
            "latitude": site["latitude"],
            "longitude": site["longitude"],
            "start_date": start.date().isoformat(),
            "end_date": end.date().isoformat(),
            "hourly": ",".join(weather["variables"]),
            "timezone": "UTC",
        },
    )
    path = raw_dir / WEATHER_FILE
    parse_openmeteo(payload).to_csv(path)
    return path


def download_prices(config: dict[str, Any], raw_dir: Path) -> Path:
    """Download day-ahead prices for the bidding zone (Energy-Charts / SMARD)."""
    start, end = _period(config)
    market = config["market"]
    payload = _fetch_json(
        f"{market['url']}/price",
        {
            "bzn": market["bidding_zone"],
            "start": start.date().isoformat(),
            "end": end.date().isoformat(),
        },
    )
    path = raw_dir / PRICE_FILE
    parse_price(payload).to_csv(path)
    return path


def download_generation(config: dict[str, Any], raw_dir: Path) -> Path:
    """Download national electricity generation by source (Energy-Charts).

    Needed to compute the grid carbon intensity with `carbon_intensity`.
    """
    start, end = _period(config)
    market = config["market"]
    chunks = []
    # One request per month, with a pause, to go easy on the service.
    for month_start in pd.date_range(start.replace(day=1), end, freq="MS"):
        chunk_start = max(start, month_start)
        chunk_end = min(end, month_start + pd.offsets.MonthEnd(0))
        payload = _fetch_json(
            f"{market['url']}/public_power",
            {
                "country": market["country"],
                "start": chunk_start.date().isoformat(),
                "end": chunk_end.date().isoformat(),
            },
        )
        chunks.append(parse_generation(payload))
        time.sleep(1)
    generation = pd.concat(chunks)
    generation = generation[~generation.index.duplicated()].sort_index()
    path = raw_dir / GENERATION_FILE
    generation.to_csv(path)
    return path


DOWNLOADS = {
    LOAD_FILE: download_load,
    PV_FILE: download_pv,
    WEATHER_FILE: download_weather,
    PRICE_FILE: download_prices,
    GENERATION_FILE: download_generation,
}


def download_all(config: dict[str, Any] | None = None, force: bool = False) -> list[Path]:
    """Download every series into `data/raw/`; skip files already there unless `force`."""
    config = config or load_config()
    raw_dir = resolve_path(config["paths"]["raw"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename, download in DOWNLOADS.items():
        path = raw_dir / filename
        if path.exists() and not force:
            print(f"already there  {path.name}")
        else:
            print(f"downloading    {path.name} ...", flush=True)
            path = download(config, raw_dir)
        paths.append(path)
    return paths


def main() -> None:
    """Download the raw data (`python -m microgrid.data`)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="download again even if the file exists")
    parser.add_argument("--config", default=None, help="path to a configuration file")
    args = parser.parse_args()
    download_all(load_config(args.config), force=args.force)


if __name__ == "__main__":
    main()
