import matplotlib as mpl

mpl.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

RESULT_DIR = Path(r"D:\Bearing training\结果\shape_loss试验")

CARD_BG = "#fbfaf7"
GRID_COLOR = "#d8dee6"
TEXT_COLOR = "#14202b"
MUTED_COLOR = "#5f6b76"

CONFIG_COLORS = {
    "baseline": "#64748b",
    "shape_v1": "#2563eb",
    "shape_v2": "#dc2626",
}

ALL_CONFIGS = ["baseline", "shape_v1", "shape_v2"]


def style_axes(ax):
    ax.set_facecolor(CARD_BG)
    ax.grid(axis="x", alpha=0.0)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.55)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c9d1da")
    ax.spines["bottom"].set_color("#c9d1da")
    ax.tick_params(colors=TEXT_COLOR, labelsize=10)


scan_df = pd.read_csv(RESULT_DIR / "详细结果_各折.csv")
summary_df = pd.read_csv(RESULT_DIR / "汇总_各配置对比.csv")

cases = sorted(scan_df["test_case"].unique())
n_configs = len(ALL_CONFIGS)
n_cases = len(cases)
bar_width = 0.8 / n_configs
x = np.arange(n_cases, dtype=float)

fig, ax = plt.subplots(figsize=(18, 6.5))
fig.patch.set_facecolor("#f4f1eb")
ax.set_facecolor(CARD_BG)

for i, config_name in enumerate(ALL_CONFIGS):
    config_data = scan_df[scan_df["config"] == config_name]
    vals = [float(config_data.loc[config_data["test_case"] == c, "life_abs_error"].values[0]) for c in cases]
    color = CONFIG_COLORS[config_name]
    offset = (i - n_configs / 2 + 0.5) * bar_width
    bars = ax.bar(x + offset, vals, width=bar_width, color=color, edgecolor="#ffffff", linewidth=0.5, label=config_name, alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2.0, val, f"{val:.0f}", ha="center", va="bottom", fontsize=7.5)

ax.set_xticks(x)
ax.set_xticklabels(cases, rotation=20, fontsize=9)
ax.set_ylabel("寿命绝对误差 (cycles)", fontsize=11, color=TEXT_COLOR)
ax.set_title("Shape Loss 试验：各工况寿命误差对比", fontsize=13, fontweight="bold", color=TEXT_COLOR, pad=12)
ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.55)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False, ncol=n_configs, loc="upper center", fontsize=9)
fig.tight_layout()
fig.savefig(RESULT_DIR / "图1_各工况寿命误差对比.png", dpi=300)
plt.close(fig)
print("Saved 图1_各工况寿命误差对比.png")

configs = ALL_CONFIGS
colors = [CONFIG_COLORS[c] for c in configs]

fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
fig.patch.set_facecolor("#f4f1eb")
metric_specs = [
    ("mean_pressure_mae", "平均压力 MAE (MPa)", "{:.2f}"),
    ("mean_wear_mae_um", "平均磨损曲线 MAE (μm)", "{:.3f}"),
    ("mean_life_abs_error", "平均寿命绝对误差 (cycles)", "{:.0f}"),
]

for ax, (column, title, fmt) in zip(axes, metric_specs):
    vals = [float(summary_df.loc[summary_df["config"] == c, column].values[0]) for c in configs]
    bars = ax.barh(configs, vals, color=colors, edgecolor="#ffffff", linewidth=1.2, height=0.68)
    style_axes(ax)
    ax.set_title(title, fontsize=12, fontweight="bold", color=TEXT_COLOR, pad=12)
    ax.set_xlabel("越低越好", fontsize=9.5, color=MUTED_COLOR, labelpad=8)
    ax.invert_yaxis()
    x_max = max(vals) * 1.18
    ax.set_xlim(0.0, x_max)

    best_val = min(vals)
    for idx, (bar, val) in enumerate(zip(bars, vals)):
        fontweight = "bold" if abs(val - best_val) < 1e-6 else "normal"
        ax.text(
            bar.get_width() + x_max * 0.015,
            bar.get_y() + bar.get_height() / 2.0,
            fmt.format(val),
            va="center",
            ha="left",
            fontsize=9.2,
            color=TEXT_COLOR,
            fontweight=fontweight,
        )

fig.suptitle("Shape Loss 试验：配置对比", fontsize=14, fontweight="bold", color=TEXT_COLOR, y=0.98)
fig.tight_layout(rect=(0.02, 0.05, 0.98, 0.93))
fig.savefig(RESULT_DIR / "图2_三指标对比.png", dpi=300)
plt.close(fig)
print("Saved 图2_三指标对比.png")
