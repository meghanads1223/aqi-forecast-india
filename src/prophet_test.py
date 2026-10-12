import logging
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

CITY = "Delhi"

df = pd.read_csv("data/processed/aqi_clean.csv", parse_dates=["Date"])
city = (
    df[df["City"] == CITY][["Date", "AQI"]]
    .rename(columns={"Date": "ds", "AQI": "y"})
)

model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
)
model.fit(city)

future = model.make_future_dataframe(periods=3)   # next 3 days
forecast = model.predict(future)

print(f"\nLast data date: {city['ds'].max().date()}")
print("Next 3 days forecast:")
print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(3).round(1))

fig = model.plot(forecast)
plt.title(f"Prophet forecast - {CITY}")
fig.savefig("models/prophet_test_delhi.png", dpi=150)
print("\nSaved plot to models/prophet_test_delhi.png")