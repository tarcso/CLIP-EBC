import matplotlib.pyplot as plt
import numpy as np

models = [
    "Teacher\n@1×",
    "Teacher\n@2×",
    "Teacher\n@4×",
    "Student\n@2×\n(fixed)",
    "Student\n@4×",
    "Student\n@2×-4×\n(scale jitter)",
    "Bicubic SR\n@2×→1×",
]

mae  = [34.49, 52.78, 109.09, 102.65, 95.73, 80.54, 45.55]
rmse = [79.71, 288.49, 566.58, 230.52, 339.87, 166.70, 168.94]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(13, 6))

bars_mae  = ax.bar(x - width/2, mae,  width, label="MAE",  color="#4C72B0")
bars_rmse = ax.bar(x + width/2, rmse, width, label="RMSE", color="#DD4949")

for bar in bars_mae:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 4,
            f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

for bar in bars_rmse:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 4,
            f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

ax.set_ylabel("Count Error", fontsize=11)
ax.set_title("NWPU validation error across resolution experiments", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=9)
ax.legend(fontsize=10)
ax.set_ylim(0, 640)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("assets/results_chart.png", dpi=200, bbox_inches="tight")
print("Saved assets/results_chart.png")
