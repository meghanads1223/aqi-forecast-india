import pandas as pd
import numpy as np

RAW = "data/raw/city_day.csv"
OUT = "data/processed/aqi_clean.csv"
SUMMARY = "data/processed/city_summary.csv"

COLS = ["City", "Date", "AQI", "PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
NUM_COLS = ["AQI", "PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
MAX_GAP = 3          # only fill gaps of up to 3 missing days
FULL_TIER_DAYS = 600 # cities with at least this many days get the LSTM too


def cpcb_category(aqi):
    """CPCB AQI categories."""
    bins = [-np.inf, 50, 100, 200, 300, 400, np.inf]
    labels = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    return pd.cut(aqi, bins=bins, labels=labels)


df = pd.read_csv(RAW, parse_dates=["Date"])

# Sanity check: does our category rule match the original AQI_Bucket column?
check = df.dropna(subset=["AQI", "AQI_Bucket"])
match = (cpcb_category(check["AQI"]).astype(str) == check["AQI_Bucket"]).mean()
print(f"Category rule matches original AQI_Bucket in {match:.1%} of rows")

df = df[COLS].drop_duplicates(subset=["City", "Date"])

frames = []
for city, g in df.groupby("City"):
    g = g.sort_values("Date").set_index("Date")

    # keep only the period where this city actually has AQI values
    first, last = g["AQI"].first_valid_index(), g["AQI"].last_valid_index()
    g = g.loc[first:last].asfreq("D")        # one row per day
    g["City"] = city

    # fill only SHORT gaps; long gaps stay empty (we don't invent data)
    g[NUM_COLS] = g[NUM_COLS].interpolate(
        method="linear", limit=MAX_GAP, limit_area="inside"
    )
    frames.append(g.reset_index())

clean = pd.concat(frames, ignore_index=True)
clean = clean.dropna(subset=["AQI"])
clean["AQI_Category"] = cpcb_category(clean["AQI"]).astype(str)

clean.to_csv(OUT, index=False)

summary = (
    clean.groupby("City")
    .agg(days=("AQI", "size"), start=("Date", "min"), end=("Date", "max"),
         avg_aqi=("AQI", "mean"))
    .round(1)
    .sort_values("days", ascending=False)
    .reset_index()
)
summary["tier"] = np.where(summary["days"] >= FULL_TIER_DAYS, "full", "limited")
summary.to_csv(SUMMARY, index=False)

print("\nCleaned rows:", len(clean))
print("\nCity summary:")
print(summary.to_string(index=False))
print("\nCities per tier:")
print(summary["tier"].value_counts())