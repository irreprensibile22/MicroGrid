"""Previsione di carico e fotovoltaico per il giorno D+1.

Uso:

    python -m microgrid.forecast

Per ciascuna delle due grandezze:

1. sceglie gli iperparametri di LightGBM con una cross-validation a finestre
   espandibili sui dati di addestramento;
2. prevede validazione e test riaddestrando il modello ogni
   `forecast.refit_every_days` giorni con tutte le misure precedenti
   (gli iperparametri restano quelli scelti al punto 1);
3. per confronto, prevede gli stessi periodi con un modello statico,
   addestrato una sola volta prima dell'inizio di ciascun periodo;
4. confronta ogni previsione con quella ingenua.

Scrive i modelli in `models/`, le previsioni in
`data/processed/forecasts.parquet` e le metriche in `results/`.
"""

from __future__ import annotations

import itertools
import json
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from microgrid.config import load_config, resolve_path
from microgrid.dataset import load_dataset, local_days
from microgrid.features import build_features

FORECAST_FILE = "forecasts.parquet"
METRICS_FILE = "forecast_metrics.csv"
PARAMS_FILE = "forecast_params.json"
# Grandezza da prevedere -> (nome breve, ritardo in ore della previsione ingenua).
TARGETS = {"load_kw": ("load", 168), "pv_kw": ("pv", 24)}


# --- Previsione ingenua e metriche -------------------------------------------


def naive_forecast(series: pd.Series, valid: pd.Series, lag_hours: int) -> pd.Series:
    """Previsione ingenua: il valore misurato `lag_hours` ore prima.

    Manca (NaN) se il valore di partenza appartiene a un giorno non valido.
    """
    source = series.where(valid)
    values = source.reindex(series.index - pd.Timedelta(hours=lag_hours)).to_numpy()
    return pd.Series(values, index=series.index)


def metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """MAE, RMSE e MAE normalizzato sulla media dei valori reali."""
    error = predicted - actual
    mae = float(error.abs().mean())
    return {
        "mae": mae,
        "rmse": float(np.sqrt((error**2).mean())),
        "nmae": mae / float(actual.mean()),
    }


# --- Cross-validation ---------------------------------------------------------


def expanding_folds(days: pd.Index, n_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Finestre espandibili per giorni interi.

    I giorni sono divisi in `n_folds + 1` blocchi consecutivi. La finestra
    `i` addestra sui primi `i + 1` blocchi e valida sul blocco successivo:
    la validazione viene sempre dopo l'addestramento.
    """
    unique_days = np.sort(pd.unique(days))
    blocks = np.array_split(unique_days, n_folds + 1)
    folds = []
    for i in range(n_folds):
        train_days = np.concatenate(blocks[: i + 1])
        folds.append((np.isin(days, train_days), np.isin(days, blocks[i + 1])))
    return folds


def _lgb_params(config: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    return {
        **config["forecast"]["base_params"],
        **params,
        "seed": config["seed"],
        "deterministic": True,
        "force_row_wise": True,
        "verbosity": -1,
    }


def tune(features: pd.DataFrame, target: pd.Series, days: pd.Index, config: dict[str, Any]) -> dict[str, Any]:
    """Sceglie gli iperparametri con il MAE medio sulle finestre espandibili.

    Per ogni combinazione, il numero di alberi è quello a cui l'errore di
    validazione smette di migliorare, mediato sulle finestre.
    """
    settings = config["forecast"]
    folds = expanding_folds(days, settings["cv_folds"])
    names = list(settings["grid"])
    trials = []
    for values in itertools.product(*(settings["grid"][name] for name in names)):
        params = dict(zip(names, values))
        errors, rounds = [], []
        for train_mask, validation_mask in folds:
            train_set = lgb.Dataset(features[train_mask], target[train_mask])
            validation_set = lgb.Dataset(features[validation_mask], target[validation_mask], reference=train_set)
            model = lgb.train(
                {**_lgb_params(config, params), "metric": "l1"},
                train_set,
                num_boost_round=settings["max_boosting_rounds"],
                valid_sets=[validation_set],
                callbacks=[lgb.early_stopping(settings["early_stopping_rounds"], verbose=False)],
            )
            errors.append(model.best_score["valid_0"]["l1"])
            rounds.append(model.best_iteration)
        trials.append({"params": params, "cv_mae": float(np.mean(errors)), "rounds": int(round(np.mean(rounds)))})
    best = min(trials, key=lambda trial: trial["cv_mae"])
    return {**best, "trials": trials}


def fit(features: pd.DataFrame, target: pd.Series, config: dict[str, Any], tuned: dict[str, Any]) -> lgb.Booster:
    """Addestra un modello con gli iperparametri e il numero di alberi scelti."""
    return lgb.train(_lgb_params(config, tuned["params"]), lgb.Dataset(features, target), num_boost_round=tuned["rounds"])


def predict(model: lgb.Booster, features: pd.DataFrame, upper: float | None = None) -> pd.Series:
    """Previsione limitata ai valori fisicamente possibili (non negativa, sotto `upper`)."""
    values = np.clip(model.predict(features), 0, upper)
    return pd.Series(values, index=features.index)


def walk_forward(
    features: pd.DataFrame,
    target: pd.Series,
    trainable: pd.Series,
    days: pd.Index,
    predict_mask: pd.Series,
    config: dict[str, Any],
    tuned: dict[str, Any],
    upper: float | None = None,
) -> tuple[pd.Series, lgb.Booster]:
    """Prevede le ore di `predict_mask` riaddestrando il modello a intervalli regolari.

    I giorni da prevedere sono divisi in blocchi di
    `forecast.refit_every_days` giorni. Per ogni blocco il modello è
    addestrato sulle sole ore utilizzabili (`trainable`) dei giorni
    precedenti l'inizio del blocco. Restituisce le previsioni e l'ultimo
    modello addestrato.
    """
    step = config["forecast"]["refit_every_days"]
    predict_days = np.sort(pd.unique(days[predict_mask.to_numpy()]))
    forecast = pd.Series(np.nan, index=features.index)
    for start in range(0, len(predict_days), step):
        block_days = predict_days[start : start + step]
        history = (trainable & (days < block_days[0])).to_numpy()
        block = predict_mask.to_numpy() & np.isin(days, block_days)
        model = fit(features[history], target[history], config, tuned)
        forecast[block] = predict(model, features[block], upper)
    return forecast, model


# --- Pipeline completo --------------------------------------------------------


def run(config: dict[str, Any] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Addestra i modelli e restituisce previsioni, metriche e iperparametri scelti.

    Colonne delle previsioni, per `load` e `pv`:

    - `*_forecast`         modello riaddestrato periodicamente (quello usato dal sistema)
    - `*_forecast_static`  modello statico: addestrato sul solo addestramento per le
                           ore di validazione, su addestramento + validazione per il test
    - `*_naive`            previsione ingenua
    """
    config = config or load_config()
    dataset = load_dataset(config)
    days = local_days(dataset.index, config)
    split = dataset["split"]
    models_dir = resolve_path(config["paths"]["models"])
    models_dir.mkdir(parents=True, exist_ok=True)

    forecasts = dataset[["split", "valid_day"]].copy()
    rows, chosen = [], {}
    for target, (name, naive_lag) in TARGETS.items():
        # Solo il carico ha giorni con misure non attendibili.
        valid = dataset["valid_day"] if target == "load_kw" else pd.Series(True, index=dataset.index)
        features = build_features(dataset, config, target)
        actual = dataset[target]
        upper = config["pv"]["peak_kw"] if target == "pv_kw" else None
        trainable = valid & actual.notna()
        train_mask = trainable & (split == "train")
        development_mask = trainable & (split != "test")

        tuned = tune(features[train_mask], actual[train_mask], days[train_mask], config)
        chosen[target] = tuned
        print(f"{target}: {tuned['params']}, {tuned['rounds']} alberi, MAE in CV {tuned['cv_mae']:.3f} kW", flush=True)

        evaluation = split != "train"
        forecast, last_model = walk_forward(features, actual, trainable, days, evaluation, config, tuned, upper)
        last_model.save_model(models_dir / f"{name}.txt")

        static = pd.Series(np.nan, index=dataset.index)
        for split_name, history in (("validation", train_mask), ("test", development_mask)):
            model = fit(features[history], actual[history], config, tuned)
            static[split == split_name] = predict(model, features[split == split_name], upper)
        naive = naive_forecast(actual, valid, naive_lag)
        if target == "pv_kw":
            # Senza irraggiamento previsto non c'è produzione.
            dark = dataset["radiation_w_m2"] <= 0
            forecast = forecast.mask(dark & forecast.notna(), 0.0)
            static = static.mask(dark & static.notna(), 0.0)

        forecasts[target] = actual
        forecasts[f"{name}_forecast"] = forecast
        forecasts[f"{name}_forecast_static"] = static
        forecasts[f"{name}_naive"] = naive
        candidates = (
            ("ingenua", naive),
            ("LightGBM statico", static),
            (f"LightGBM riaddestrato ogni {config['forecast']['refit_every_days']} giorni", forecast),
        )
        for split_name in ("validation", "test"):
            # Confronto sulle stesse ore: quelle in cui esistono tutte le previsioni.
            rows_mask = (split == split_name) & valid & forecast.notna() & naive.notna()
            for model_name, predicted in candidates:
                rows.append(
                    {
                        "target": name,
                        "split": split_name,
                        "model": model_name,
                        "hours": int(rows_mask.sum()),
                        **metrics(actual[rows_mask], predicted[rows_mask]),
                    }
                )
    forecasts = forecasts[forecasts["split"] != "train"]
    return forecasts, pd.DataFrame(rows), chosen


def load_forecasts(config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Legge le previsioni salvate da `python -m microgrid.forecast`."""
    config = config or load_config()
    path = resolve_path(config["paths"]["processed"]) / FORECAST_FILE
    if not path.exists():
        raise FileNotFoundError(f"Manca {path}: esegui prima `python -m microgrid.forecast`.")
    return pd.read_parquet(path)


def main() -> None:
    config = load_config()
    forecasts, table, chosen = run(config)
    results_dir = resolve_path(config["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)
    forecasts.to_parquet(resolve_path(config["paths"]["processed"]) / FORECAST_FILE)
    table.to_csv(results_dir / METRICS_FILE, index=False, float_format="%.4f")
    (results_dir / PARAMS_FILE).write_text(json.dumps(chosen, indent=2), encoding="utf-8")
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
