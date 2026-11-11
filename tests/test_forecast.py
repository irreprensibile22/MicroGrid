"""Test di feature e previsione, con attenzione all'uso di informazioni future."""

import numpy as np
import pandas as pd
import pytest

from microgrid import forecast
from microgrid.config import load_config
from microgrid.dataset import local_days
from microgrid.features import build_features, calendar_features, day_starts, history_features, lag_features
from microgrid.forecast import expanding_folds, metrics, naive_forecast, walk_forward

CONFIG = load_config()
WEATHER_FEATURES = ["temperature_c", "radiation_w_m2", "cloud_cover_pct", "temperature_day_mean"]


def make_dataset(start: str, days: int, seed: int = 0) -> pd.DataFrame:
    """Dataset sintetico con le colonne usate dalle feature."""
    timezone = CONFIG["site"]["timezone"]
    first = pd.Timestamp(start, tz=timezone)
    index = pd.date_range(first, first + pd.Timedelta(days=days), freq="h", inclusive="left").tz_convert("UTC")
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "load_kw": rng.uniform(1, 10, len(index)),
            "pv_kw": rng.uniform(0, 50, len(index)),
            "temperature_c": rng.uniform(-5, 25, len(index)),
            "radiation_w_m2": rng.uniform(0, 800, len(index)),
            "cloud_cover_pct": rng.uniform(0, 100, len(index)),
            "valid_day": True,
        },
        index=index,
    )


def test_day_starts_is_local_midnight():
    index = pd.DatetimeIndex([pd.Timestamp("2016-07-01 21:00", tz="UTC"), pd.Timestamp("2016-07-01 22:00", tz="UTC")])
    starts = day_starts(index, CONFIG)
    # D'estate la mezzanotte di Costanza è alle 22:00 UTC del giorno prima.
    assert starts[0] == pd.Timestamp("2016-06-30 22:00", tz="UTC")
    assert starts[1] == pd.Timestamp("2016-07-01 22:00", tz="UTC")


def test_calendar_features_mark_weekends_and_holidays():
    index = pd.DatetimeIndex(
        [pd.Timestamp(day + " 12:00", tz="Europe/Berlin") for day in ("2016-01-06", "2016-01-07", "2016-01-09")]
    ).tz_convert("UTC")
    features = calendar_features(index, CONFIG)
    # Il 6 gennaio è festivo nel Baden-Württemberg; il 9 gennaio 2016 è un sabato.
    assert features["is_holiday"].tolist() == [1, 0, 0]
    assert features["is_weekend"].tolist() == [0, 0, 1]
    assert features["hour"].tolist() == [12, 12, 12]
    assert "day_of_week" not in calendar_features(index, CONFIG, weekday=False)


def test_lag_features_never_use_values_measured_after_issue_time():
    # Il 30 ottobre 2016 ha 25 ore: la sua ultima ora dista 24 ore dalla prima.
    dataset = make_dataset("2016-10-28", 5)
    index = dataset.index
    issue_time = day_starts(index, CONFIG)
    lags = lag_features(dataset["load_kw"], dataset["valid_day"], [24], issue_time)
    long_day = local_days(index, CONFIG) == pd.Timestamp("2016-10-30")
    assert long_day.sum() == 25
    assert lags.loc[long_day, "lag_24h"].isna().sum() == 1
    assert np.isnan(lags.loc[long_day, "lag_24h"].iloc[-1])
    # In un giorno normale il ritardo di 24 ore è il valore del giorno prima.
    position = 30
    assert lags["lag_24h"].iloc[position] == dataset["load_kw"].iloc[position - 24]


def test_lag_and_history_features_skip_invalid_days():
    dataset = make_dataset("2016-05-01", 10)
    days = local_days(dataset.index, CONFIG)
    dataset.loc[days == pd.Timestamp("2016-05-03"), "valid_day"] = False
    lags = lag_features(dataset["load_kw"], dataset["valid_day"], [24], day_starts(dataset.index, CONFIG))
    history = history_features(dataset["load_kw"], dataset["valid_day"], days, min_days=1)
    next_day = days == pd.Timestamp("2016-05-04")
    assert lags.loc[next_day, "lag_24h"].isna().all()
    assert history.loc[next_day, "mean_previous_day"].isna().all()
    # Il 5 maggio usa il 4, che è valido.
    following = days == pd.Timestamp("2016-05-05")
    expected = dataset.loc[next_day, "load_kw"].mean()
    assert history.loc[following, "mean_previous_day"].to_numpy() == pytest.approx(expected)


@pytest.mark.parametrize("target", ["load_kw", "pv_kw"])
def test_features_of_a_day_do_not_depend_on_that_day_or_later(target):
    dataset = make_dataset("2016-10-20", 14)
    days = local_days(dataset.index, CONFIG)
    day = pd.Timestamp("2016-10-30")
    changed = dataset.copy()
    changed.loc[days >= day, target] = changed.loc[days >= day, target] + 1000.0
    before = build_features(dataset, CONFIG, target)
    after = build_features(changed, CONFIG, target)
    pd.testing.assert_frame_equal(before[days == day], after[days == day])
    # Controprova: modificare il giorno precedente cambia le feature.
    changed = dataset.copy()
    changed.loc[days == day - pd.Timedelta(days=1), target] += 1000.0
    assert not build_features(changed, CONFIG, target)[days == day].equals(before[days == day])


def test_naive_forecast_repeats_past_value_and_skips_invalid_days():
    index = pd.date_range("2016-01-01", periods=72, freq="h", tz="UTC")
    series = pd.Series(np.arange(72.0), index=index)
    valid = pd.Series(True, index=index)
    valid.iloc[:24] = False
    naive = naive_forecast(series, valid, 24)
    assert naive.iloc[:48].isna().all()
    assert naive.iloc[48:].tolist() == series.iloc[24:48].tolist()


def test_metrics_values():
    actual = pd.Series([10.0, 10.0, 20.0, 20.0])
    predicted = pd.Series([12.0, 8.0, 20.0, 24.0])
    result = metrics(actual, predicted)
    assert result["mae"] == pytest.approx(2.0)
    assert result["rmse"] == pytest.approx(np.sqrt(6.0))
    assert result["nmae"] == pytest.approx(2.0 / 15.0)


def test_expanding_folds_validate_only_on_later_days():
    days = pd.Index(pd.date_range("2016-01-01", periods=50, freq="D").repeat(24))
    folds = expanding_folds(days, 4)
    assert len(folds) == 4
    previous_train = 0
    for train_mask, validation_mask in folds:
        assert not (train_mask & validation_mask).any()
        assert days[train_mask].max() < days[validation_mask].min()
        assert train_mask.sum() > previous_train
        previous_train = train_mask.sum()


def test_walk_forward_trains_only_on_days_before_each_block(monkeypatch):
    dataset = make_dataset("2016-03-01", 30)
    days = local_days(dataset.index, CONFIG)
    features = build_features(dataset, CONFIG, "load_kw")
    target = dataset["load_kw"]
    predict_mask = pd.Series(days >= pd.Timestamp("2016-03-21"), index=dataset.index)
    trainable = pd.Series(True, index=dataset.index)
    seen = []

    class ConstantModel:
        def __init__(self, value):
            self.value = value

        def predict(self, rows):
            return np.full(len(rows), self.value)

    def fake_fit(train_features, train_target, config, tuned):
        seen.append(local_days(train_features.index, CONFIG).max())
        return ConstantModel(train_target.mean())

    monkeypatch.setattr(forecast, "fit", fake_fit)
    config = {**CONFIG, "forecast": {**CONFIG["forecast"], "refit_every_days": 4}}
    predicted, _ = walk_forward(features, target, trainable, days, predict_mask, config, tuned={})
    # 10 giorni da prevedere a blocchi di 4: tre addestramenti, ognuno fermo al giorno prima del blocco.
    assert seen == [pd.Timestamp("2016-03-20"), pd.Timestamp("2016-03-24"), pd.Timestamp("2016-03-28")]
    assert predicted[predict_mask].notna().all()
    assert predicted[~predict_mask].isna().all()
