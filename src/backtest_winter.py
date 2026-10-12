import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import logging
import numpy as np
import pandas as pd
from prophet import Prophet
from tensorflow import keras
from tensorflow.keras import layers

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)
keras.utils.set_random_seed(42)

DATA = "data/processed/aqi_clean.csv"
SUMMARY = "data/processed/city_summary.csv"
OUT = "data/processed/backtest_winter.csv"

CUTOFF = pd.Timestamp("2019-11-15")
N_ORIGINS = 20          # every 3 days -> 60-day window
FEATURES = ["AQI", "PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
LOOKBACK = 14
HORIZON = 3
VAL_DAYS = 45
MIN_TRAIN_ROWS = 365

df = pd.read_csv(DATA, parse_dates=["Date"])
summary = pd.read_csv(SUMMARY)
FULL = summary.loc[summary["tier"] == "full", "City"].tolist()


def city_frame(city):
    g = df[df["City"] == city].sort_values("Date").set_index("Date")
    full_index = pd.date_range(g.index.min(), g.index.max(), freq="D")
    return g.reindex(full_index)[FEATURES]


def compute_stats(cdf, cutoff):
    part = cdf.loc[:cutoff, FEATURES]
    mean = np.nan_to_num(part.mean().values)
    std = part.std().values
    std = np.where(np.isnan(std) | (std < 1e-6), 1.0, std)
    return mean, std


def make_windows(cdf, mean, std):
    z = ((cdf[FEATURES] - mean) / std).values
    z = np.nan_to_num(z, nan=0.0)
    aqi = cdf["AQI"].values
    doy = cdf.index.dayofyear.values
    cal = np.stack([np.sin(2 * np.pi * doy / 365.25),
                    np.cos(2 * np.pi * doy / 365.25)], axis=1)
    full = np.concatenate([z, cal], axis=1)
    isn = np.isnan(aqi)

    X, Y, origins, last = [], [], [], []
    for t in range(LOOKBACK - 1, len(cdf)):
        if isn[t] or isn[t - LOOKBACK + 1: t + 1].sum() > 3:
            continue
        if t + HORIZON < len(cdf):
            y = (aqi[t + 1: t + 1 + HORIZON] - aqi[t]) / std[0]
        else:
            y = np.full(HORIZON, np.nan)
        X.append(full[t - LOOKBACK + 1: t + 1])
        Y.append(y)
        origins.append(cdf.index[t])
        last.append(aqi[t])
    return (np.array(X, dtype="float32"), np.array(Y, dtype="float32"),
            pd.DatetimeIndex(origins), np.array(last))


# ---------- prepare cities ----------
frames, stats, windows, skipped = {}, {}, {}, []
Xtr, Ytr, Xva, Yva = [], [], [], []

for city in FULL:
    cdf = city_frame(city)
    if cdf.index.max() < CUTOFF + pd.Timedelta(days=60) or \
            cdf.loc[:CUTOFF, "AQI"].notna().sum() < MIN_TRAIN_ROWS:
        skipped.append(city)
        continue
    mean, std = compute_stats(cdf, CUTOFF)
    X, Y, o, last = make_windows(cdf, mean, std)
    frames[city] = cdf
    stats[city] = {"mean": mean.tolist(), "std": std.tolist()}
    windows[city] = (X, Y, o, last)

    ok = np.isfinite(Y).all(axis=1)
    tr_end = CUTOFF - pd.Timedelta(days=HORIZON)
    val_start = tr_end - pd.Timedelta(days=VAL_DAYS)
    tr = ok & (o <= val_start)
    va = ok & (o > val_start) & (o <= tr_end)
    Xtr.append(X[tr]); Ytr.append(Y[tr])
    Xva.append(X[va]); Yva.append(Y[va])

if skipped:
    print("Skipped (not enough data for this window):", ", ".join(skipped))

Xtr, Ytr = np.concatenate(Xtr), np.concatenate(Ytr)
Xva, Yva = np.concatenate(Xva), np.concatenate(Yva)
print(f"Winter window starts {CUTOFF.date()} | cities: {len(frames)}")
print(f"Training windows: {len(Xtr)} | Validation windows: {len(Xva)}")

# ---------- LSTM ----------
model = keras.Sequential([
    layers.Input(shape=(LOOKBACK, Xtr.shape[2])),
    layers.LSTM(64),
    layers.Dropout(0.2),
    layers.Dense(32, activation="relu"),
    layers.Dense(HORIZON),
])
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="huber")
early = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=8, restore_best_weights=True
)
model.fit(Xtr, Ytr, validation_data=(Xva, Yva),
          epochs=60, batch_size=64, callbacks=[early], verbose=2)

origins = pd.date_range(CUTOFF, periods=N_ORIGINS, freq="3D")
lstm_rows = []
for city in frames:
    X, Y, o, last = windows[city]
    sel = np.isin(o, origins)
    if sel.sum() == 0:
        continue
    pred = model.predict(X[sel], verbose=0)
    std_aqi = stats[city]["std"][0]
    for i, origin in enumerate(o[sel]):
        for h in range(HORIZON):
            val = np.clip(last[sel][i] + pred[i, h] * std_aqi, 0, 500)
            lstm_rows.append({"City": city, "origin": origin,
                              "horizon": h + 1, "lstm": float(val)})
lstm_df = pd.DataFrame(lstm_rows)

# ---------- Prophet + persistence ----------
rows = []
for city, cdf in frames.items():
    s = cdf["AQI"]
    tr = s.loc[:CUTOFF].dropna()
    train = pd.DataFrame({"ds": tr.index, "y": tr.values})
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                daily_seasonality=False)
    m.fit(train)
    future = pd.DataFrame({"ds": pd.date_range(
        CUTOFF + pd.Timedelta(days=1), periods=N_ORIGINS * 3, freq="D")})
    fc = m.predict(future).set_index("ds")["yhat"]

    for origin in origins:
        last_val = s.get(origin, np.nan)
        if pd.isna(last_val):
            continue
        for h in range(1, HORIZON + 1):
            d = origin + pd.Timedelta(days=h)
            actual = s.get(d, np.nan)
            if pd.isna(actual) or d not in fc.index:
                continue
            rows.append({"City": city, "origin": origin, "horizon": h,
                         "date": d, "actual": float(actual),
                         "prophet": float(np.clip(fc[d], 0, 500)),
                         "persistence": float(last_val)})
    print("Prophet done:", city)

bt = pd.DataFrame(rows)
m = bt.merge(lstm_df, on=["City", "origin", "horizon"], how="inner")
m.to_csv(OUT, index=False)


def mae(a, b): return float(np.mean(np.abs(a - b)))
def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))


MODELS = ["prophet", "persistence", "lstm"]
print(f"\nEvaluated on {len(m)} forecasts across {m['City'].nunique()} cities")

print("\nOverall (same rows for all models):")
for name in MODELS:
    print(f"{name:12s} MAE {mae(m['actual'], m[name]):6.1f}   "
          f"RMSE {rmse(m['actual'], m[name]):6.1f}")

print("\nMAE by forecast day:")
for h, g in m.groupby("horizon"):
    parts = " | ".join(f"{n} {mae(g['actual'], g[n]):.1f}" for n in MODELS)
    print(f"Day {h}: {parts}")

print("\nPer-city MAE:")
table = []
for city, g in m.groupby("City"):
    table.append({"City": city, **{n: mae(g["actual"], g[n]) for n in MODELS}})
table = pd.DataFrame(table).round(1)
print(table.to_string(index=False))
print("\nCities where LSTM beats persistence (MAE):",
      int((table["lstm"] < table["persistence"]).sum()), "of", len(table))