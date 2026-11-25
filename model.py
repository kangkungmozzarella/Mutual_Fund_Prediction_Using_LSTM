"""Model LSTM untuk prediksi harga Close, dengan parameter yang bisa diatur dari dashboard.

Perbaikan dibanding notebook skripsi:
- Target benar-benar kolom Close (di notebook, target = kolom pertama fitur = High).
- Scaler di-fit hanya pada data latih (menghindari kebocoran data uji).
- EarlyStopping memakai validasi dari ujung data latih, bukan data uji.
- Forecast rekursif mengisi semua fitur (OHLC, Average, RSI) dari hasil prediksi,
  bukan mengisi 0, sehingga tidak "mendatar" secara artifisial.
"""
import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

ALL_FEATURES = ["Close", "Open", "High", "Low", "Average", "RSI", "Volume"]
TARGET = "Close"


@dataclass
class Params:
    features: list = field(default_factory=lambda: ["Close", "Open", "High", "Low", "Average", "RSI"])
    window: int = 30          # panjang urutan (hari)
    split: float = 0.8        # porsi data latih
    layers: int = 2           # jumlah lapis LSTM
    units: int = 50           # unit per lapis
    dropout: float = 0.2
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 0.001
    patience: int = 10        # EarlyStopping
    horizon: int = 30         # hari bursa ke depan
    rsi_period: int = 14
    seed: int = 42

    LIMITS = {
        "window": (5, 120), "split": (0.5, 0.95), "layers": (1, 3), "units": (8, 256),
        "dropout": (0.0, 0.6), "epochs": (1, 300), "batch_size": (1, 256),
        "learning_rate": (1e-5, 0.05), "patience": (1, 50), "horizon": (1, 120),
        "rsi_period": (2, 50), "seed": (0, 10_000),
    }

    @classmethod
    def from_dict(cls, d: dict) -> "Params":
        p = cls()
        for name, (lo, hi) in cls.LIMITS.items():
            if name in d and d[name] is not None:
                cast = type(getattr(p, name))
                setattr(p, name, cast(min(max(cast(d[name]), lo), hi)))
        feats = [f for f in d.get("features", p.features) if f in ALL_FEATURES]
        if TARGET not in feats:
            feats.insert(0, TARGET)
        p.features = [TARGET] + [f for f in feats if f != TARGET]  # target selalu kolom 0
        return p

    def to_dict(self) -> dict:
        return asdict(self)


def add_features(df: pd.DataFrame, rsi_period: int = 14) -> pd.DataFrame:
    d = df.copy()
    d["Average"] = (d.Open + d.High + d.Low + d.Close) / 4
    delta = d.Close.diff()
    gain = delta.clip(lower=0).rolling(rsi_period).mean()
    loss = (-delta.clip(upper=0)).rolling(rsi_period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi[(loss == 0) & (gain > 0)] = 100
    d["RSI"] = rsi.fillna(50)  # awal seri & hari tanpa pergerakan dianggap netral
    if "Volume" not in d:
        d["Volume"] = 0.0
    return d


def _sequences(arr: np.ndarray, window: int):
    x = np.stack([arr[i - window:i] for i in range(window, len(arr))])
    y = arr[window:, 0]
    return x, y


def _build_model(p: Params, n_features: int):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(p.seed)
    m = tf.keras.Sequential([tf.keras.Input(shape=(p.window, n_features))])
    for i in range(p.layers):
        m.add(tf.keras.layers.LSTM(p.units, return_sequences=i < p.layers - 1))
        m.add(tf.keras.layers.Dropout(p.dropout))
    m.add(tf.keras.layers.Dense(1))
    m.compile(optimizer=tf.keras.optimizers.Adam(p.learning_rate), loss="mse")
    return m


def _metrics(actual: np.ndarray, pred: np.ndarray, prev: np.ndarray) -> dict:
    err = actual - pred
    changed = actual != prev
    hit = np.sign(pred - prev)[changed] == np.sign(actual - prev)[changed]
    ss_res, ss_tot = float((err ** 2).sum()), float(((actual - actual.mean()) ** 2).sum())
    return {
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mae": float(np.abs(err).mean()),
        "mape": float((np.abs(err) / actual).mean() * 100),
        "r2": 1 - ss_res / ss_tot if ss_tot else None,
        "dir_acc": float(hit.mean() * 100) if hit.size else None,
        "dir_hit": int(hit.sum()), "dir_miss": int((~hit).sum()),
    }


def train_and_forecast(df: pd.DataFrame, p: Params, on_epoch=None) -> dict:
    """Latih model, evaluasi di data uji, lalu forecast `horizon` hari bursa ke depan."""
    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler

    t0 = time.time()
    data = add_features(df, p.rsi_period)
    values = data[p.features].to_numpy(dtype="float64")
    n = len(values)
    n_train = int(n * p.split)
    if n_train <= p.window + 10 or n - n_train < 5:
        raise ValueError(f"Data terlalu sedikit ({n} baris) untuk window {p.window} dan split {p.split}")

    scaler = MinMaxScaler().fit(values[:n_train])
    scaled = scaler.transform(values)
    x_train, y_train = _sequences(scaled[:n_train], p.window)
    x_test, y_test = _sequences(scaled[n_train - p.window:], p.window)

    tmin, trange = scaler.data_min_[0], scaler.data_range_[0]
    unscale = lambda v: np.asarray(v).reshape(-1) * trange + tmin

    class Progress(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if on_epoch:
                logs = logs or {}
                on_epoch(epoch + 1, p.epochs, float(logs.get("loss", math.nan)),
                         float(logs.get("val_loss", math.nan)))

    model = _build_model(p, len(p.features))
    hist = model.fit(
        x_train, y_train, epochs=p.epochs, batch_size=p.batch_size,
        validation_split=0.1, shuffle=False, verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=p.patience,
                                                    restore_best_weights=True), Progress()],
    )

    # --- Evaluasi data uji ---
    pred = unscale(model.predict(x_test, verbose=0))
    actual = unscale(y_test)
    test_dates = data.index[n_train:]
    prev = data.Close.to_numpy()[n_train - 1:-1]  # harga kemarin = tebakan naive
    m = _metrics(actual, pred, prev)
    naive = _metrics(actual, prev, prev)
    m["naive_rmse"], m["naive_mape"] = naive["rmse"], naive["mape"]
    m["vs_naive_pct"] = (1 - m["rmse"] / naive["rmse"]) * 100 if naive["rmse"] else None

    # --- Forecast rekursif: setiap prediksi dijadikan baris OHLC baru, fitur dihitung ulang ---
    raw = df[["Open", "High", "Low", "Close"] + (["Volume"] if "Volume" in df else [])].copy()
    future_dates = pd.bdate_range(raw.index[-1] + pd.Timedelta(days=1), periods=p.horizon)
    fc = []
    for d in future_dates:
        feat = add_features(raw, p.rsi_period)[p.features].to_numpy(dtype="float64")
        window = scaler.transform(feat[-p.window:])[None, ...]
        close = float(unscale(model.predict(window, verbose=0))[0])
        last_close = float(raw.Close.iloc[-1])
        row = {"Open": last_close, "High": max(last_close, close), "Low": min(last_close, close),
               "Close": close}
        if "Volume" in raw:
            row["Volume"] = float(raw.Volume.tail(20).mean())
        raw.loc[d] = row
        fc.append(close)

    # Rentang perkiraan: sebaran error data uji (robust terhadap pencilan, pakai MAD),
    # melebar seiring horizon
    resid = actual - pred
    sigma = float(1.4826 * np.median(np.abs(resid - np.median(resid)))) or float(np.std(resid))
    band = [1.96 * sigma * math.sqrt(h) for h in range(1, p.horizon + 1)]

    r = lambda v: None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), 4)
    return {
        "params": p.to_dict(),
        "metrics": {k: r(v) if isinstance(v, float) else v for k, v in m.items()},
        "loss": [{"epoch": i + 1, "loss": r(l), "val_loss": r(v)}
                 for i, (l, v) in enumerate(zip(hist.history["loss"], hist.history.get("val_loss", [])))],
        "best_epoch": int(np.argmin(hist.history.get("val_loss", hist.history["loss"])) + 1),
        "test": [{"date": d.strftime("%Y-%m-%d"), "actual": r(a), "predicted": r(pr), "naive": r(nv)}
                 for d, a, pr, nv in zip(test_dates, actual, pred, prev)],
        "forecast": [{"date": d.strftime("%Y-%m-%d"), "predicted": r(v), "lo": r(v - b), "hi": r(v + b)}
                     for d, v, b in zip(future_dates, fc, band)],
        "rows": {"total": n, "train": n_train, "test": n - n_train},
        "seconds": round(time.time() - t0, 1),
    }
