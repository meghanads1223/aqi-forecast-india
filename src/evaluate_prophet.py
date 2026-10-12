import logging
import warnings
import numpy as np
import pandas as pd
from prophet import Prophet

logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

DATA = "data/processed/aqi_clean.csv"
SUMMARY = "data/processed/city_summary.csv"
OUT = "data/processed/backtest_prophet.csv"

TEST_DAYS = 60   # backtest window: last 60 days of each city
STEP = 3         # new forecast origin every 3 days
HORIZON = 3      # forecast 3 days ahead

df = pd.read_csv(DATA, parse_dates=["Date"])
summary = pd.read_csv(SUMMARY)

rows = []
for city in summary["City"]:
    g = (
        df[df["City"] == city][["Date", "AQI"]]
        .rename(columns={"Date": "ds", "AQI": "y"})
        .sort_values("ds")
    )
    actual = g.set_index("ds")["y"]
    last = g["ds"].max()
    origins = pd.date_range(
        last - pd.Timedelta(days=TEST_DAYS),
        last - pd.Timedelta(days=HORIZON),
        freq=f"{STEP}D",
    )

    for origin in origins:
        train = g[g["ds"] <= origin]
        if len(train) < 30:
            continue
        model = Prophet(
            yearly_seasonality="auto",
            weekly_seasonality=True,
            daily_seasonality=False,
        )
        model.fit(train)

        future = pd.DataFrame(
            {"ds": pd.date_range(origin + pd.Timedelta(days=1), periods=HORIZON)}
        )
        pred = np.clip(model.predict(future)["yhat"].values, 0, 500)
        last_value = train["y"].iloc[-1]

        for h, (d, p) in enumerate(zip(future["ds"], pred), start=1):
            if d in actual.index:
                rows.append({
                    "City": city, "origin": origin, "horizon": h, "date": d,
                    "actual": actual[d], "prophet": p, "persistence": last_value,
                })
    print("done:", city, flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT, index=False)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


per_city = []
for city, g in res.groupby("City"):
    per_city.append({
        "City": city, "n": len(g),
        "Prophet_MAE": mae(g["actual"], g["prophet"]),
        "Persist_MAE": mae(g["actual"], g["persistence"]),
        "Prophet_RMSE": rmse(g["actual"], g["prophet"]),
        "Persist_RMSE": rmse(g["actual"], g["persistence"]),
    })
per_city = pd.DataFrame(per_city).round(1)

print("\nPer-city results:")
print(per_city.to_string(index=False))

print("\nOverall:")
print(f"Prophet     MAE {mae(res['actual'], res['prophet']):.1f}  "
      f"RMSE {rmse(res['actual'], res['prophet']):.1f}")
print(f"Persistence MAE {mae(res['actual'], res['persistence']):.1f}  "
      f"RMSE {rmse(res['actual'], res['persistence']):.1f}")

print("\nMAE by forecast day:")
for h, g in res.groupby("horizon"):
    print(f"Day {h}: Prophet {mae(g['actual'], g['prophet']):.1f} | "
          f"Persistence {mae(g['actual'], g['persistence']):.1f}")