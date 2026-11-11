"""Download delle serie orarie grezze (carico, fotovoltaico, meteo, prezzi, CO2).

Uso:

    python -m microgrid.data            # scarica ciò che manca in data/raw/
    python -m microgrid.data --force    # riscarica tutto

Ogni funzione `download_*` salva un file in `data/raw/` e ne restituisce il
percorso. Tutti gli istanti sono in UTC e indicano l'inizio dell'intervallo.
La pulizia e l'unione delle serie avvengono in un passo successivo.
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
# Margine attorno al periodo di studio, per coprire la conversione fra ora
# locale e UTC e le feature ritardate dei primi giorni.
MARGIN = pd.Timedelta(days=1)

LOAD_FILE = "opsd_household_60min.csv"
PV_FILE = "pvgis_hourly_per_kwp.csv"
WEATHER_FILE = "openmeteo_hourly.csv"
PRICE_FILE = "day_ahead_price.csv"
GENERATION_FILE = "generation_by_type.csv"


# --- Rete -------------------------------------------------------------------


def _fetch(url: str, params: dict[str, Any] | None = None, retries: int = 6) -> bytes:
    """Scarica un URL, riprovando con attesa crescente in caso di errore temporaneo."""
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    wait = 5.0
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            # 4xx diversi da 429 sono errori della richiesta: riprovare non serve.
            if error.code < 500 and error.code != 429:
                raise
            last_error: Exception = error
            # Con 429 il servizio indica quanti secondi aspettare.
            retry_after = error.headers.get("Retry-After", "")
            wait = float(retry_after) + 1 if retry_after.isdigit() else wait
        except (urllib.error.URLError, TimeoutError) as error:
            last_error = error
        if attempt < retries - 1:
            time.sleep(wait)
            wait = 5 * 2 ** (attempt + 1)
    raise RuntimeError(f"Download fallito dopo {retries} tentativi: {url}") from last_error


def _fetch_json(url: str, params: dict[str, Any] | None = None) -> Any:
    return json.loads(_fetch(url, params))


def _period(config: dict[str, Any]) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Estremi (date, con margine) dell'intervallo da scaricare."""
    start = pd.Timestamp(config["data"]["start"]) - MARGIN
    end = pd.Timestamp(config["data"]["end"]) + MARGIN
    return start, end


# --- Conversione delle risposte (funzioni pure, testabili senza rete) --------


def parse_pvgis(payload: dict[str, Any]) -> pd.DataFrame:
    """Converte la risposta oraria di PVGIS in potenza per kWp installato.

    PVGIS indica gli istanti come `YYYYMMDD:HHMM` in UTC, con i minuti del
    passaggio del satellite (es. `:10`): il valore viene attribuito all'ora
    che contiene quell'istante. La potenza `P` è in W.
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
    """Converte la risposta oraria di Open-Meteo (richiesta in UTC) in tabella."""
    hourly = pd.DataFrame(payload["hourly"])
    index = pd.to_datetime(hourly.pop("time"), utc=True)
    hourly.index = pd.DatetimeIndex(index, name="utc_timestamp")
    return hourly


def parse_price(payload: dict[str, Any]) -> pd.DataFrame:
    """Converte la risposta `/price` di Energy-Charts in prezzi [EUR/MWh]."""
    if payload.get("unit") != "EUR / MWh":
        raise ValueError(f"Unità di prezzo inattesa: {payload.get('unit')!r}")
    index = pd.to_datetime(payload["unix_seconds"], unit="s", utc=True)
    return pd.DataFrame(
        {"price_eur_mwh": payload["price"]},
        index=pd.DatetimeIndex(index, name="utc_timestamp"),
    )


def parse_generation(payload: dict[str, Any]) -> pd.DataFrame:
    """Converte la risposta `/public_power` di Energy-Charts in potenza [MW] per fonte."""
    index = pd.to_datetime(payload["unix_seconds"], unit="s", utc=True)
    columns = {series["name"]: series["data"] for series in payload["production_types"]}
    return pd.DataFrame(columns, index=pd.DatetimeIndex(index, name="utc_timestamp"))


def carbon_intensity(generation: pd.DataFrame, factors: dict[str, float]) -> pd.Series:
    """Intensità di carbonio della produzione [gCO2eq/kWh].

    È la media dei fattori di emissione pesata sulla potenza prodotta da
    ciascuna fonte. Si considerano solo le fonti presenti in `factors`;
    la produzione negativa (es. consumo dei pompaggi) è trattata come zero.
    Non tiene conto degli scambi con l'estero.
    """
    missing = sorted(set(factors) - set(generation.columns))
    if missing:
        raise KeyError(f"Fonti senza dati di produzione: {missing}")
    power = generation[list(factors)].clip(lower=0)
    weights = pd.Series(factors, dtype=float)
    total = power.sum(axis=1, min_count=1)
    intensity = power.mul(weights, axis=1).sum(axis=1, min_count=1) / total
    return intensity.rename("carbon_intensity_g_kwh")


# --- Download ---------------------------------------------------------------


def download_load(config: dict[str, Any], raw_dir: Path) -> Path:
    """Scarica le misure orarie delle utenze di Costanza (Open Power System Data).

    Il file è salvato così com'è: contiene letture cumulative di contatore
    [kWh] per ogni utenza e apparecchio.
    """
    path = raw_dir / LOAD_FILE
    path.write_bytes(_fetch(config["load"]["source_url"]))
    return path


def download_pv(config: dict[str, Any], raw_dir: Path) -> Path:
    """Scarica la produzione fotovoltaica simulata da PVGIS, per 1 kWp installato.

    Salvare la serie per kWp permette di cambiare la taglia dell'impianto
    senza riscaricare.
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
    """Scarica il meteo storico orario da Open-Meteo (rianalisi)."""
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
    """Scarica i prezzi day-ahead della zona di mercato (Energy-Charts / SMARD)."""
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
    """Scarica la produzione elettrica nazionale per fonte (Energy-Charts).

    Serve a calcolare l'intensità di carbonio della rete con
    `carbon_intensity`. La richiesta è divisa per mese per non
    sovraccaricare il servizio.
    """
    start, end = _period(config)
    market = config["market"]
    chunks = []
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
    """Scarica tutte le serie in `data/raw/`; salta i file già presenti se non `force`."""
    config = config or load_config()
    raw_dir = resolve_path(config["paths"]["raw"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename, download in DOWNLOADS.items():
        path = raw_dir / filename
        if path.exists() and not force:
            print(f"già presente  {path.name}")
        else:
            print(f"scarico       {path.name} ...", flush=True)
            path = download(config, raw_dir)
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="riscarica anche i file già presenti")
    parser.add_argument("--config", default=None, help="percorso di un file di configurazione")
    args = parser.parse_args()
    download_all(load_config(args.config), force=args.force)


if __name__ == "__main__":
    main()
