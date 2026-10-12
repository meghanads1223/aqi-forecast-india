import pandas as pd

df = pd.read_csv("data/raw/city_day.csv", parse_dates=["Date"])

print("Rows, columns:", df.shape)
print("Date range:", df["Date"].min().date(), "to", df["Date"].max().date())
print("Number of cities:", df["City"].nunique())

print("\nRows with an AQI value, per city:")
print(df.dropna(subset=["AQI"]).groupby("City").size().sort_values(ascending=False))

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nAQI_Bucket counts:")
print(df["AQI_Bucket"].value_counts())