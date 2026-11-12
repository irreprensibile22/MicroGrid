"""Load and PV forecast for day D+1.

Usage:

    python -m microgrid.forecast

For each of the two targets:

1. pick the LightGBM hyperparameters with an expanding-window
   cross-validation on the training data;
2. forecast validation and test, refitting the model every
   `forecast.refit_every_days` days on all measurements available so far
   (hyperparameters stay those chosen in step 1);
3. for comparison, forecast the same periods with a static model, trained
   once before each period starts;
4. compare every forecast with the naive one.

Writes the models to `models/`, the forecasts to
`data/processed/forecasts.parquet` and the metrics to `results/`.
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
# Target -> (short name, lag in hours of the naive forecast).
TARGETS = {"load_kw": ("load", 168), "pv_kw": ("pv", 24)}
NAIVE = "naive"
STATIC = "LightGBM static"


def refit_label(config: dict[str, Any]) -> str:
    """Name of the periodically refitted model in tables and charts."""
    return f"LightGBM refit every {config['forecast']['refit_every_days']} days"


# --- Naive forecast and metrics ----------------------------------------------


def naive_forecast(series: pd.Series, valid: pd.Series, lag_hours: int) -> pd.Series:
    """Naive forecast: the value measured `lag_hours` earlier.

    Missing (NaN) if the source value belongs to an invalid day.
    """
    source = series.where(valid)
    values = source.reindex(series.index - pd.Timedelta(hours=lag_hours)).to_numpy()
    return pd.Series(values, index=series.index)


def metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """MAE, RMSE and MAE normalised by the mean of the actual values."""
    error = predicted - actual
    mae = float(error.abs().mean())
    return {
        "mae": mae,
        "rmse": float(np.sqrt((error**2).mean())),
        "nmae": mae / float(actual.mean()),
    }


# --- Cross-validation --------------------------------------------------------


def expanding_folds(days: pd.Index, n_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding windows over whole days.

    The days are cut into `n_folds + 1` consecutive blocks. Fold `i` trains on
    the first `i + 1` blocks and validates on the next one, so validation
    always comes after training.
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
        # Fixed seed + deterministic mode: the same data always give the same model.
        "seed": config["seed"],
        "deterministic": True,
        "force_row_wise": True,
        "verbosity": -1,
    }


def tune(features: pd.DataFrame, target: pd.Series, days: pd.Index, config: dict[str, Any]) -> dict[str, Any]:
    """Pick the hyperparameters with the lowest mean MAE over the expanding windows.

    For each combination, the number of trees is where the validation error
    stops improving, averaged over the folds.
    """
    settings = config["forecast"]
    folds = expanding_folds(days, settings["cv_folds"])
    names = list(settings["grid"])
    trials = []
    for values in itertools.product(*(settings["grid"][name] for name in names)):
        params = dict(zip(names, values, strict=True))
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
    """Train a model with the chosen hyperparameters and number of trees."""
    return lgb.train(
        _lgb_params(config, tuned["params"]), lgb.Dataset(features, target), num_boost_round=tuned["rounds"]
    )


def predict(model: lgb.Booster, features: pd.DataFrame, upper: float | None = None) -> pd.Series:
    """Forecast clipped to what is physically possible (non-negative, below `upper`)."""
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
    """Forecast the hours in `predict_mask`, refitting the model at regular intervals.

    The days to forecast are cut into blocks of `forecast.refit_every_days`
    days. For each block the model is trained only on the usable hours
    (`trainable`) of the days before the block starts. Returns the forecasts
    and the last model trained.
    """
    step = config["forecast"]["refit_every_days"]
    predict_days = np.sort(pd.unique(days[predict_mask.to_numpy()]))
    forecast = pd.Series(np.nan, index=features.index)
    for start in range(0, len(predict_days), step):
        block_days = predict_days[start : start + step]
        # Strictly before the block: this is what keeps the walk-forward honest.
        history = (trainable & (days < block_days[0])).to_numpy()
        block = predict_mask.to_numpy() & np.isin(days, block_days)
        model = fit(features[history], target[history], config, tuned)
        forecast[block] = predict(model, features[block], upper)
    return forecast, model


# --- Full pipeline -----------------------------------------------------------


def run(config: dict[str, Any] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Train the models and return forecasts, metrics and chosen hyperparameters.

    Forecast columns, for `load` and `pv`:

    - `*_forecast`         periodically refitted model (the one the system uses)
    - `*_forecast_static`  static model: trained on train only for the validation
                           hours, on train + validation for the test hours
    - `*_naive`            naive forecast
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
        # Only the load has days with unreliable measurements.
        valid = dataset["valid_day"] if target == "load_kw" else pd.Series(True, index=dataset.index)
        features = build_features(dataset, config, target)
        actual = dataset[target]
        upper = config["pv"]["peak_kw"] if target == "pv_kw" else None
        trainable = valid & actual.notna()
        train_mask = trainable & (split == "train")
        development_mask = trainable & (split != "test")

        tuned = tune(features[train_mask], actual[train_mask], days[train_mask], config)
        chosen[target] = tuned
        print(f"{target}: {tuned['params']}, {tuned['rounds']} trees, CV MAE {tuned['cv_mae']:.3f} kW", flush=True)

        evaluation = split != "train"
        forecast, last_model = walk_forward(features, actual, trainable, days, evaluation, config, tuned, upper)
        last_model.save_model(models_dir / f"{name}.txt")

        static = pd.Series(np.nan, index=dataset.index)
        for split_name, history in (("validation", train_mask), ("test", development_mask)):
            model = fit(features[history], actual[history], config, tuned)
            static[split == split_name] = predict(model, features[split == split_name], upper)
        naive = naive_forecast(actual, valid, naive_lag)
        if target == "pv_kw":
            # No forecast irradiance, no output.
            dark = dataset["radiation_w_m2"] <= 0
            forecast = forecast.mask(dark & forecast.notna(), 0.0)
            static = static.mask(dark & static.notna(), 0.0)

        forecasts[target] = actual
        forecasts[f"{name}_forecast"] = forecast
        forecasts[f"{name}_forecast_static"] = static
        forecasts[f"{name}_naive"] = naive
        candidates = ((NAIVE, naive), (STATIC, static), (refit_label(config), forecast))
        for split_name in ("validation", "test"):
            # Compare on the same hours: those where every forecast exists.
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
    """Read the forecasts saved by `python -m microgrid.forecast`."""
    config = config or load_config()
    path = resolve_path(config["paths"]["processed"]) / FORECAST_FILE
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}: run `python -m microgrid.forecast` first.")
    return pd.read_parquet(path)


def main() -> None:
    """Train the models and save forecasts and metrics (`python -m microgrid.forecast`)."""
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
