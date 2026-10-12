import os
import pandas as pd
import matplotlib.pyplot as plt

FILES = [
    ("Calm window (last 60 days)", "data/processed/backtest_all.csv"),
    ("Winter window (from 2019-11-15)", "data/processed/backtest_winter.csv"),
]
MODELS = ["prophet", "persistence", "lstm"]
LABELS = {"prophet": "Prophet", "persistence": "Persistence", "lstm": "LSTM"}
COLORS = {"prophet": "#9aa0a6", "persistence": "#4c78a8", "lstm": "#e45756"}
OUT = "results/mae_by_forecast_day.png"

os.makedirs("results", exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)

for ax, (title, path) in zip(axes, FILES):
    m = pd.read_csv(path)
    mae = (
        m.assign(**{n: (m["actual"] - m[n]).abs() for n in MODELS})
        .groupby("horizon")[MODELS]
        .mean()
    )
    width = 0.26
    x = range(len(mae))
    for i, n in enumerate(MODELS):
        pos = [p + (i - 1) * width for p in x]
        bars = ax.bar(pos, mae[n], width, label=LABELS[n], color=COLORS[n])
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                    f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"Day {h}" for h in mae.index])
    ax.set_ylabel("MAE (AQI points)")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)

axes[0].legend(frameon=False)
fig.suptitle("Forecast error by horizon, 18 Indian cities (lower is better)")
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print("Saved", OUT)