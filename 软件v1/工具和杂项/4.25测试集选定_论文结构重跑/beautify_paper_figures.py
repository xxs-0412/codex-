from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\28382\Desktop\Research on the Wear Life Prediction of Coated Spherical Plain Bearings\软件v1")
RESULT = ROOT / "结果" / "4.25测试集选定_论文结构重跑"
OUT = RESULT / "论文图表_美化版"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 140,
    "savefig.dpi": 360,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "axes.labelcolor": "#172033",
    "xtick.color": "#172033",
    "ytick.color": "#172033",
    "text.color": "#172033",
})

COLORS = {
    "FNN": "#4E79A7",
    "GRU": "#59A14F",
    "LSTM": "#B07AA1",
    "1D-CNN": "#F28E2B",
    "Transformer": "#E15759",
    "Enhanced Transformer": "#0F766E",
    "T0_baseline": "#94A3B8",
    "T1_module1_only": "#2563EB",
    "T2_module2_only": "#F97316",
    "T3_module1_plus_module2": "#0F766E",
}

DISPLAY = {
    "T0_baseline": "T0 Baseline",
    "T1_module1_only": "T1 Module 1",
    "T2_module2_only": "T2 Module 2",
    "T3_module1_plus_module2": "T3 Module 1+2",
    "M1_0_5d_baseline": "5D baseline",
    "M1_R1_static_ratio": "R1 static ratios",
    "M1_R2_log_cycle_replace": "R2 log-cycle",
    "M1_R3_log_cycle_keep": "R3 cycle + log-cycle",
    "M1_R2_log_cycle_replace__S1_slow_abs_0p01": "R2 + slow 0.01",
    "M1_R2_log_cycle_replace__S1_slow_abs_0p003": "R2 + slow 0.003",
    "M1_R2_log_cycle_replace__S1_slow_abs_0p03": "R2 + slow 0.03",
    "M1_R3_log_cycle_keep__S1_slow_abs_0p003": "R3 + slow 0.003",
    "M1_R3_log_cycle_keep__S1_slow_abs_0p01": "R3 + slow 0.01",
    "M1_R3_log_cycle_keep__S1_slow_abs_0p03": "R3 + slow 0.03",
    "M1_R2_log_cycle_replace__S2_temporal_mono_plus_slow_abs_0p003": "R2 + mono+slow 0.003",
    "M1_R2_log_cycle_replace__S2_temporal_mono_plus_slow_abs_0p01": "R2 + mono+slow 0.01",
    "M1_R2_log_cycle_replace__S2_temporal_mono_plus_slow_abs_0p03": "R2 + mono+slow 0.03",
}


def read_csv(rel: str) -> pd.DataFrame:
    return pd.read_csv(RESULT / rel)


def label(x: str) -> str:
    return DISPLAY.get(str(x), str(x))


def style(ax) -> None:
    ax.grid(axis="x", color="#CBD5E1", alpha=0.55, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def mean_std_bar(df: pd.DataFrame, title: str, name: str, label_col="model", sort=True, highlight=None) -> None:
    plot = df.copy()
    if sort:
        plot = plot.sort_values("mean_life_abs_error", ascending=True)
    plot["display"] = plot[label_col].map(label)
    y = np.arange(len(plot))
    vals = plot["mean_life_abs_error"].astype(float).to_numpy()
    errs = plot.get("std_life_abs_error", pd.Series(np.zeros(len(plot)))).astype(float).to_numpy()
    colors = [COLORS.get(str(v), "#64748B") for v in plot[label_col].astype(str)]
    if highlight is not None:
        colors = ["#0F766E" if str(v) == highlight else "#94A3B8" for v in plot[label_col].astype(str)]

    fig, ax = plt.subplots(figsize=(10.8, max(4.8, 0.62 * len(plot) + 1.8)), constrained_layout=True)
    ax.barh(y, vals, xerr=errs, color=colors, edgecolor="white", linewidth=1.1, capsize=4, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(plot["display"], fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlabel("Mean life absolute error (cycles), mean ± std", fontsize=11)
    ax.set_title(title, fontsize=14, pad=12)
    style(ax)
    xmax = float(np.nanmax(vals + errs)) * 1.22
    ax.set_xlim(0, xmax)
    for yi, v, e in zip(y, vals, errs):
        ax.text(v + e + xmax * 0.015, yi, f"{v:.0f} ± {e:.0f}", va="center", ha="left", fontsize=9.5)
    save(fig, name)


def module_screen_chart(df: pd.DataFrame, title: str, name: str, top_n: int | None = None) -> None:
    plot = df.sort_values("mean_life_abs_error", ascending=True).copy()
    if top_n is not None:
        plot = plot.head(top_n)
    plot["display"] = plot["variant"].map(label)
    fig, ax = plt.subplots(figsize=(11.8, max(5.2, 0.55 * len(plot) + 1.7)), constrained_layout=True)
    y = np.arange(len(plot))
    vals = plot["mean_life_abs_error"].astype(float).to_numpy()
    errs = plot.get("std_life_abs_error", pd.Series(np.zeros(len(plot)))).astype(float).to_numpy()
    ax.barh(y, vals, xerr=errs, color="#2563EB", alpha=0.86, edgecolor="white", linewidth=1.0, capsize=4)
    ax.set_yticks(y)
    ax.set_yticklabels(plot["display"], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("Mean life absolute error (cycles), mean ± std", fontsize=11)
    ax.set_title(title, fontsize=14, pad=12)
    style(ax)
    xmax = float(np.nanmax(vals + errs)) * 1.2
    ax.set_xlim(0, xmax)
    for yi, v, e in zip(y, vals, errs):
        ax.text(v + e + xmax * 0.012, yi, f"{v:.0f}", va="center", ha="left", fontsize=9)
    save(fig, name)


def ablation_chart(df: pd.DataFrame) -> None:
    order = ["T0_baseline", "T1_module1_only", "T2_module2_only", "T3_module1_plus_module2"]
    plot = df.set_index("variant").loc[order].reset_index()
    plot["display"] = plot["variant"].map(label)
    x = np.arange(len(plot))
    vals = plot["mean_life_abs_error"].astype(float).to_numpy()
    errs = plot["std_life_abs_error"].astype(float).to_numpy()
    colors = [COLORS.get(v, "#64748B") for v in plot["variant"]]
    fig, ax = plt.subplots(figsize=(9.6, 5.8), constrained_layout=True)
    bars = ax.bar(x, vals, yerr=errs, color=colors, edgecolor="white", linewidth=1.2, capsize=5, width=0.62)
    ax.plot(x, vals, color="#111827", marker="o", linewidth=1.4, alpha=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels(plot["display"], fontsize=10, rotation=10, ha="right")
    ax.set_ylabel("Mean life absolute error (cycles)", fontsize=11)
    ax.set_title("Ablation Study of the Enhanced Transformer", fontsize=14, pad=12)
    ax.grid(axis="y", color="#CBD5E1", alpha=0.55)
    ax.set_axisbelow(True)
    ymax = float(np.nanmax(vals + errs)) * 1.22
    ax.set_ylim(0, ymax)
    for bar, v, e in zip(bars, vals, errs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + e + ymax * 0.018, f"{v:.0f}\n±{e:.0f}", ha="center", va="bottom", fontsize=9)
    save(fig, "Fig4_ablation_mean_std")


def per_run_heatmap(detail: pd.DataFrame) -> None:
    plot = detail.copy()
    plot["display"] = plot["model"].map(label)
    pivot = plot.pivot_table(index="display", columns="test_case", values="life_abs_error", aggfunc="mean")
    order = [label(x) for x in ["Enhanced Transformer", "LSTM", "GRU", "Transformer", "1D-CNN", "FNN"] if label(x) in pivot.index]
    pivot = pivot.loc[order]
    fig, ax = plt.subplots(figsize=(8.8, 5.3), constrained_layout=True)
    im = ax.imshow(pivot.values, cmap="YlGnBu_r", aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(c).replace(".xlsx", "") for c in pivot.columns], fontsize=10)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)
    ax.set_title("Per-Run Life Error Heatmap", fontsize=14, pad=12)
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("Life absolute error (cycles)")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=9, color="#111827")
    save(fig, "Fig6_final_per_run_error_heatmap")


def main() -> None:
    baseline = read_csv(r"01_五模型基准对比\汇总_各模型平均指标.csv")
    m1_screen = read_csv(r"02_模块一_物理派生特征比较\粗筛\汇总_各模型平均指标.csv")
    m1_refine = read_csv(r"02_模块一_物理派生特征比较\复筛\汇总_各模型平均指标.csv")
    m2_screen = read_csv(r"03_模块二_趋势约束比较\粗筛\汇总_各模型平均指标.csv")
    m2_refine = read_csv(r"03_模块二_趋势约束比较\复筛\汇总_各模型平均指标.csv")
    ablation = read_csv(r"04_最终消融实验\汇总_各模型平均指标.csv")
    final = read_csv(r"05_最终横向对比\汇总_各模型平均指标.csv")
    final_detail = read_csv(r"05_最终横向对比\详细结果_各模型各测试集.csv")

    mean_std_bar(baseline, "Five-Model Baseline Comparison", "Fig1_baseline_model_comparison")
    module_screen_chart(m1_screen, "Module 1 Feature Engineering: Coarse Screening", "Fig2a_module1_coarse_screen")
    module_screen_chart(m1_refine, "Module 1 Feature Engineering: Refined Top-2", "Fig2b_module1_refine_top2")
    module_screen_chart(m2_screen, "Module 2 Trend Constraint: Coarse Screening", "Fig3a_module2_coarse_screen", top_n=10)
    module_screen_chart(m2_refine, "Module 2 Trend Constraint: Refined Candidates", "Fig3b_module2_refine")
    ablation_chart(ablation)
    mean_std_bar(final, "Final Comparison: Baselines vs Enhanced Transformer", "Fig5_final_comparison", highlight="Enhanced Transformer")
    per_run_heatmap(final_detail)

    appendix = json.loads((RESULT / "05_最终横向对比" / "追加seed判断.json").read_text(encoding="utf-8"))
    best_m2 = json.loads((RESULT / "03_模块二_趋势约束比较" / "模块二最佳组合.json").read_text(encoding="utf-8"))
    top2 = json.loads((RESULT / "02_模块一_物理派生特征比较" / "模块一_top2.json").read_text(encoding="utf-8"))
    lines = [
        "图表检查与美化说明",
        "=" * 72,
        "一、检查结论",
        "- 原始结果图已生成，但日志中出现 tight_layout 排版警告，说明部分图存在标题/坐标轴/标签空间不足风险。",
        "- 为避免论文插图出现文字拥挤、图例压住数据、长标签重叠等问题，已基于正式 CSV 重新生成 `论文图表_美化版`。",
        "- 新图统一使用更大的画布、统一配色、mean ± std 误差棒、较少但清晰的数值标注，并同时输出 PNG 与 SVG。",
        "- 中文字体优先使用 Microsoft YaHei / SimHei；美化版标题和坐标轴多采用英文短标签，进一步降低乱码风险。",
        "",
        "二、推荐用于论文的图",
        "- Fig1_baseline_model_comparison：五模型基准对比。",
        "- Fig2b_module1_refine_top2：模块一 top2 复筛对比。",
        "- Fig3b_module2_refine：模块二复筛对比。",
        "- Fig4_ablation_mean_std：最终四格消融实验。",
        "- Fig5_final_comparison：最终横向对比。",
        "- Fig6_final_per_run_error_heatmap：各测试工况误差热力图，可作为补充图或分析图。",
        "",
        "三、关键实验结论快照",
        f"- 模块一 top2: {', '.join(top2)}。",
        f"- 模块二最佳组合: {best_m2.get('best_variant')}；特征={best_m2.get('best_feature')}；shape={best_m2.get('best_shape')}。",
        f"- 最终横向对比最优: {final.sort_values('mean_life_abs_error').iloc[0]['variant']}，mean_life_abs_error={float(final.sort_values('mean_life_abs_error').iloc[0]['mean_life_abs_error']):.1f}。",
        f"- 追加 seed 判断: needs_10_seed_extension={appendix.get('needs_10_seed_extension')}；advantage={float(appendix.get('advantage')):.1f}；threshold={float(appendix.get('threshold')):.1f}。",
        "",
        "四、文件位置",
        f"- 美化图目录: {OUT}",
    ]
    (RESULT / "图表检查与美化说明.txt").write_text("\n".join(lines), encoding="utf-8")
    print("beautified figures written to", OUT)


if __name__ == "__main__":
    main()
