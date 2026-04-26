from __future__ import annotations

import argparse
import ctypes
from pathlib import Path
import tkinter as tk
from tkinter import font as tkfont
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from predict_life import (
    available_coatings,
    default_request,
    load_request,
    project_root,
    run_prediction_request,
    save_prediction_artifacts,
    timestamp_stem,
)


FIELD_SPECS = [
    ("F", "载荷 F (N)", 4325.0),
    ("D", "内径 D (mm)", 10.0),
    ("Cr", "径向间隙 Cr (mm)", 0.0500),
    ("elastic_modulus_GPa", "弹性模量 E (GPa)", 210.0),
    ("actual_cycle_step", "递推步长 (实际转数)", 1672.0),
    ("wear_threshold_um", "磨损阈值 (um)", 5.0),
]


class PredictionGUI:
    def __init__(self) -> None:
        self.root_dir = project_root()
        self.input_path = self.root_dir / "app_workspace" / "prediction_input.csv"
        self.colors = {
            "bg": "#eef2f4",
            "surface": "#ffffff",
            "surface_alt": "#f6f8fa",
            "hero": "#13324b",
            "hero_text": "#f7fbff",
            "accent": "#d97706",
            "accent_dark": "#b45309",
            "accent_soft": "#fff3df",
            "success": "#1f7a4d",
            "success_soft": "#e7f5ee",
            "warning": "#b45309",
            "warning_soft": "#fff4e5",
            "info": "#2563eb",
            "info_soft": "#eaf2ff",
            "text": "#14202b",
            "muted": "#5f6b76",
            "line": "#d7dde2",
        }

        self._configure_windows_dpi()
        self.root = tk.Tk()
        self.root.title("涂层型关节轴承寿命预测软件")
        self.root.geometry("1440x960")
        self.root.minsize(1200, 860)
        self.root.configure(bg=self.colors["bg"] )
        self._apply_tk_scaling()
        self.fonts = {
            "base": tkfont.Font(family="Microsoft YaHei UI", size=10),
            "small": tkfont.Font(family="Microsoft YaHei UI", size=9),
            "label": tkfont.Font(family="Microsoft YaHei UI", size=11, weight="bold"),
            "hero_title": tkfont.Font(family="Microsoft YaHei UI", size=20, weight="bold"),
            "hero_subtitle": tkfont.Font(family="Microsoft YaHei UI", size=10),
            "metric_value": tkfont.Font(family="Segoe UI", size=19, weight="bold"),
            "metric_title": tkfont.Font(family="Microsoft YaHei UI", size=9),
            "summary": tkfont.Font(family="Microsoft YaHei UI", size=11, weight="bold"),
            "placeholder": tkfont.Font(family="Microsoft YaHei UI", size=11),
            "mono": tkfont.Font(family="Consolas", size=9),
            "button": tkfont.Font(family="Microsoft YaHei UI", size=10),
            "button_bold": tkfont.Font(family="Microsoft YaHei UI", size=10, weight="bold"),
        }

        self.entries: dict[str, tk.StringVar] = {}
        self.coating_var = tk.StringVar(value=available_coatings()[0])
        self.summary_var = tk.StringVar(value="点击“开始预测”后，这里会显示寿命总转数、阈值与当前预测模式。")
        self.warning_var = tk.StringVar(value="当前处于待预测状态。")
        self.status_var = tk.StringVar(value="等待输入参数。")
        self.recommendation_var = tk.StringVar(value="递推步长建议 500-2000 转。")
        self.life_value_var = tk.StringVar(value="--")
        self.life_sub_var = tk.StringVar(value="寿命总转数")
        self.mode_value_var = tk.StringVar(value="Idle")
        self.mode_sub_var = tk.StringVar(value="等待开始预测")
        self.save_value_var = tk.StringVar(value="Standby")
        self.save_sub_var = tk.StringVar(value="预测后只在界面显示")

        self.result_canvas: FigureCanvasTkAgg | None = None
        self.result_toolbar: NavigationToolbar2Tk | None = None
        self.latest_export_df: pd.DataFrame | None = None
        self.latest_figure = None
        self.latest_csv_path: Path | None = None
        self.latest_plot_path: Path | None = None
        self.left_canvas: tk.Canvas | None = None
        self.chart_holder: tk.Frame | None = None
        self.toolbar_holder: tk.Frame | None = None
        self.chart_shell: tk.Frame | None = None
        self.pending_render_job: str | None = None

        self._configure_style()
        self._build_layout()
        self.load_request_into_form()

    def _configure_windows_dpi(self) -> None:
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass

    def _apply_tk_scaling(self) -> None:
        try:
            dpi = ctypes.windll.user32.GetDpiForSystem()
            self.root.tk.call("tk", "scaling", dpi / 72.0)
        except Exception:
            pass
    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        self.root.option_add("*Font", self.fonts["base"])
        style.configure("App.TFrame", background=self.colors["bg"])
        style.configure(
            "Primary.TButton",
            font=self.fonts["button_bold"],
            padding=(14, 10),
            background=self.colors["accent"],
            foreground="#ffffff",
            borderwidth=0,
            focusthickness=0,
        )
        style.map(
            "Primary.TButton",
            background=[("active", self.colors["accent_dark"]), ("pressed", self.colors["accent_dark"])],
            foreground=[("disabled", "#ffffff")],
        )
        style.configure(
            "Secondary.TButton",
            font=self.fonts["button"],
            padding=(12, 9),
            background=self.colors["surface_alt"],
            foreground=self.colors["text"],
            borderwidth=1,
            focusthickness=0,
        )
        style.map(
            "Secondary.TButton",
            background=[("active", "#e8edf2"), ("pressed", "#dde4eb")],
        )
        style.configure(
            "Export.TButton",
            font=self.fonts["button"],
            padding=(12, 9),
            background="#f3f6f8",
            foreground=self.colors["text"],
            borderwidth=1,
            focusthickness=0,
        )
        style.map("Export.TButton", background=[("active", "#e4eaef")])
        style.configure("App.TEntry", padding=8)
        style.configure("App.TCombobox", padding=6)
        style.configure("App.TNotebook", background=self.colors["surface"], borderwidth=0)
        style.configure(
            "App.TNotebook.Tab",
            font=self.fonts["small"],
            padding=(12, 8),
            background=self.colors["surface_alt"],
            foreground=self.colors["muted"],
        )
        style.map(
            "App.TNotebook.Tab",
            background=[("selected", self.colors["surface"])],
            foreground=[("selected", self.colors["text"])],
        )

    def _card(
        self,
        parent: tk.Widget,
        title: str,
        subtitle: str | None = None,
        padx: int = 16,
        pady: int = 16,
        subtitle_wraplength: int = 320,
    ):
        card = tk.Frame(
            parent,
            bg=self.colors["surface"],
            highlightbackground=self.colors["line"],
            highlightthickness=1,
            bd=0,
        )
        header = tk.Frame(card, bg=self.colors["surface"])
        header.pack(fill="x", padx=padx, pady=(pady, 8))

        tk.Label(
            header,
            text=title,
            bg=self.colors["surface"],
            fg=self.colors["text"],
            font=self.fonts["label"],
        ).pack(anchor="w")
        if subtitle:
            tk.Label(
                header,
                text=subtitle,
                bg=self.colors["surface"],
                fg=self.colors["muted"],
                font=self.fonts["small"],
                wraplength=subtitle_wraplength,
                justify="left",
            ).pack(anchor="w", pady=(4, 0))

        body = tk.Frame(card, bg=self.colors["surface"])
        body.pack(fill="both", expand=True, padx=padx, pady=(0, pady))
        return card, body

    def _metric_card(self, parent: tk.Widget, title: str, value_var: tk.StringVar, subtitle_var: tk.StringVar, accent: str):
        card = tk.Frame(
            parent,
            bg=self.colors["surface"],
            highlightbackground=self.colors["line"],
            highlightthickness=1,
            bd=0,
            height=124,
        )
        card.pack_propagate(False)
        top_bar = tk.Frame(card, bg=accent, height=5)
        top_bar.pack(fill="x")
        top_bar.pack_propagate(False)

        body = tk.Frame(card, bg=self.colors["surface"])
        body.pack(fill="both", expand=True, padx=16, pady=10)

        tk.Label(
            body,
            text=title,
            bg=self.colors["surface"],
            fg=self.colors["muted"],
            font=self.fonts["metric_title"],
        ).pack(anchor="w")
        tk.Label(
            body,
            textvariable=value_var,
            bg=self.colors["surface"],
            fg=self.colors["text"],
            font=self.fonts["metric_value"],
        ).pack(anchor="w", pady=(6, 4))
        tk.Label(
            body,
            textvariable=subtitle_var,
            bg=self.colors["surface"],
            fg=self.colors["muted"],
            font=self.fonts["small"],
            justify="left",
        ).pack(anchor="w")
        return card

    def _banner(self, parent: tk.Widget, text_var: tk.StringVar, tone: str = "info"):
        bg = self.colors["info_soft"] if tone == "info" else self.colors["warning_soft"]
        fg = self.colors["info"] if tone == "info" else self.colors["warning"]
        banner = tk.Frame(parent, bg=bg, bd=0)
        label = tk.Label(
            banner,
            textvariable=text_var,
            bg=bg,
            fg=fg,
            font=self.fonts["small"],
            justify="left",
            wraplength=860,
        )
        label.pack(anchor="w", padx=12, pady=10)
        return banner, label

    def _build_layout(self) -> None:
        root_frame = tk.Frame(self.root, bg=self.colors["bg"])
        root_frame.pack(fill="both", expand=True, padx=14, pady=14)

        hero = tk.Frame(root_frame, bg=self.colors["hero"])
        hero.pack(fill="x", pady=(0, 14))

        hero_left = tk.Frame(hero, bg=self.colors["hero"])
        hero_left.pack(side="left", fill="both", expand=True, padx=22, pady=14)
        tk.Label(
            hero_left,
            text="涂层型关节轴承寿命预测软件",
            bg=self.colors["hero"],
            fg=self.colors["hero_text"],
            font=self.fonts["hero_title"],
        ).pack(anchor="w")
        tk.Label(
            hero_left,
            text="Physics-enhanced recursive wear forecasting with threshold-based life estimation",
            bg=self.colors["hero"],
            fg="#dbe7f1",
            font=self.fonts["hero_subtitle"],
        ).pack(anchor="w", pady=(6, 0))

        hero_right = tk.Frame(hero, bg=self.colors["hero"])
        hero_right.pack(side="right", padx=22, pady=14)
        tk.Label(
            hero_right,
            text="DLC COATING\nSURROGATE MODEL",
            bg=self.colors["accent_soft"],
            fg=self.colors["accent_dark"],
            font=self.fonts["button_bold"],
            justify="center",
            padx=18,
            pady=10,
        ).pack()

        content = tk.Frame(root_frame, bg=self.colors["bg"])
        content.pack(fill="both", expand=True)

        left_shell = tk.Frame(content, bg=self.colors["bg"], width=420)
        left_shell.pack(side="left", fill="y", padx=(0, 14))
        left_shell.pack_propagate(False)

        self.left_canvas = tk.Canvas(
            left_shell,
            bg=self.colors["bg"],
            highlightthickness=0,
            bd=0,
            yscrollincrement=18,
        )
        left_scrollbar = ttk.Scrollbar(left_shell, orient="vertical", command=self.left_canvas.yview)
        self.left_canvas.configure(yscrollcommand=left_scrollbar.set)
        self.left_canvas.pack(side="left", fill="both", expand=True)
        left_scrollbar.pack(side="right", fill="y")

        left_panel = tk.Frame(self.left_canvas, bg=self.colors["bg"], width=398)
        left_window = self.left_canvas.create_window((0, 0), window=left_panel, anchor="nw")

        left_panel.bind(
            "<Configure>",
            lambda _event: self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all")),
        )
        self.left_canvas.bind(
            "<Configure>",
            lambda event: self.left_canvas.itemconfigure(left_window, width=event.width),
        )
        self.left_canvas.bind("<Enter>", self._enable_left_mousewheel)
        self.left_canvas.bind("<Leave>", self._disable_left_mousewheel)
        left_panel.bind("<Enter>", self._enable_left_mousewheel)
        left_panel.bind("<Leave>", self._disable_left_mousewheel)

        right_panel = tk.Frame(content, bg=self.colors["bg"])
        right_panel.pack(side="left", fill="both", expand=True)

        self._build_left_panel(left_panel)
        self._build_right_panel(right_panel)

    def _enable_left_mousewheel(self, _event=None) -> None:
        self.root.bind_all("<MouseWheel>", self._on_left_mousewheel)

    def _disable_left_mousewheel(self, _event=None) -> None:
        self.root.unbind_all("<MouseWheel>")

    def _on_left_mousewheel(self, event) -> None:
        if self.left_canvas is None:
            return
        step = -1 if event.delta > 0 else 1
        self.left_canvas.yview_scroll(step, "units")

    def _build_left_panel(self, parent: tk.Frame) -> None:
        coating_card, coating_body = self._card(
            parent,
            "涂层选择",
            "当前软件支持通过涂层类型切换不同模型。现在默认启用 DLC。",
        )
        coating_card.pack(fill="x", pady=(0, 12))

        tk.Label(
            coating_body,
            text="涂层类型",
            bg=self.colors["surface"],
            fg=self.colors["muted"],
            font=self.fonts["small"],
        ).pack(anchor="w", pady=(0, 6))
        coating_box = ttk.Combobox(
            coating_body,
            textvariable=self.coating_var,
            values=available_coatings(),
            state="readonly",
            style="App.TCombobox",
        )
        coating_box.pack(fill="x")

        input_card, input_body = self._card(
            parent,
            "工况输入",
            "输入载荷、几何尺寸、递推步长和磨损阈值。结果会显示在右侧界面中。",
        )
        input_card.pack(fill="x", pady=(0, 12))

        grid_panel = tk.Frame(input_body, bg=self.colors["surface"])
        grid_panel.pack(fill="x")
        grid_panel.grid_columnconfigure(0, weight=1)
        grid_panel.grid_columnconfigure(1, weight=1)

        for index, (field_key, label_text, _) in enumerate(FIELD_SPECS):
            row, col = divmod(index, 2)
            field_box = tk.Frame(grid_panel, bg=self.colors["surface"])
            field_box.grid(row=row, column=col, sticky="ew", padx=(0, 8) if col == 0 else (8, 0), pady=(0, 10))
            tk.Label(
                field_box,
                text=label_text,
                bg=self.colors["surface"],
                fg=self.colors["muted"],
                font=self.fonts["small"],
            ).pack(anchor="w", pady=(0, 4))
            value_var = tk.StringVar()
            entry = ttk.Entry(field_box, textvariable=value_var, style="App.TEntry")
            entry.pack(fill="x")
            self.entries[field_key] = value_var

        note_panel = tk.Frame(input_body, bg=self.colors["info_soft"], bd=0)
        note_panel.pack(fill="x", pady=(2, 0))
        tk.Label(
            note_panel,
            textvariable=self.recommendation_var,
            bg=self.colors["info_soft"],
            fg=self.colors["info"],
            justify="left",
            wraplength=340,
            font=self.fonts["small"],
            padx=12,
            pady=8,
        ).pack(anchor="w")

        action_card, action_body = self._card(
            parent,
            "操作面板",
            "先保存参数，再开始预测。软件会自动在结果区渲染曲线。",
        )
        action_card.pack(fill="x", pady=(0, 12))

        small_button_row = tk.Frame(action_body, bg=self.colors["surface"])
        small_button_row.pack(fill="x", pady=(0, 8))
        ttk.Button(small_button_row, text="从 CSV 载入", command=self.load_request_into_form, style="Secondary.TButton").pack(
            side="left", fill="x", expand=True
        )
        ttk.Button(small_button_row, text="保存参数", command=self.save_current_request, style="Secondary.TButton").pack(
            side="left", fill="x", expand=True, padx=(8, 0)
        )
        ttk.Button(action_body, text="开始预测", command=self.run_prediction, style="Primary.TButton").pack(fill="x")

        aux_card, aux_body = self._card(
            parent,
            "导出与状态",
            "当前版本不自动保存结果；这里保留原界面的导出位置。",
        )
        aux_card.pack(fill="both", expand=True)

        notebook = ttk.Notebook(aux_body, style="App.TNotebook")
        notebook.pack(fill="both", expand=True)

        export_body = tk.Frame(notebook, bg=self.colors["surface"])
        status_body = tk.Frame(notebook, bg=self.colors["surface"])
        notebook.add(export_body, text="导出")
        notebook.add(status_body, text="状态")

        export_inner = tk.Frame(export_body, bg=self.colors["surface"])
        export_inner.pack(fill="both", expand=True, padx=4, pady=4)

        tk.Label(
            export_inner,
            text="根目录 bat 启动方式保持不变；当前预测结果只显示在界面中。",
            bg=self.colors["surface"],
            fg=self.colors["muted"],
            justify="left",
            wraplength=320,
            font=self.fonts["small"],
        ).pack(anchor="w", pady=(0, 10))
        self.save_csv_button = ttk.Button(export_inner, text="另存预测数据 CSV", command=self.save_csv_as, style="Export.TButton", state="disabled")
        self.save_csv_button.pack(fill="x")
        self.save_plot_button = ttk.Button(export_inner, text="另存曲线图片 PNG", command=self.save_plot_as, style="Export.TButton", state="disabled")
        self.save_plot_button.pack(fill="x", pady=(8, 0))

        status_inner = tk.Frame(status_body, bg=self.colors["surface"])
        status_inner.pack(fill="both", expand=True, padx=4, pady=4)

        tk.Label(
            status_inner,
            textvariable=self.status_var,
            bg=self.colors["surface"],
            fg=self.colors["muted"],
            justify="left",
            wraplength=320,
            font=self.fonts["mono"],
        ).pack(anchor="w")

    def _build_right_panel(self, parent: tk.Frame) -> None:
        metric_row = tk.Frame(parent, bg=self.colors["bg"])
        metric_row.pack(fill="x", pady=(0, 10))
        metric_row.grid_columnconfigure((0, 1, 2), weight=1)
        metric_row.grid_rowconfigure(0, minsize=124)

        self._metric_card(metric_row, "寿命总转数", self.life_value_var, self.life_sub_var, self.colors["accent"]).grid(
            row=0, column=0, sticky="nsew", padx=(0, 6)
        )
        self._metric_card(metric_row, "预测模式", self.mode_value_var, self.mode_sub_var, self.colors["info"]).grid(
            row=0, column=1, sticky="nsew", padx=6
        )
        self._metric_card(metric_row, "结果显示", self.save_value_var, self.save_sub_var, self.colors["success"]).grid(
            row=0, column=2, sticky="nsew", padx=(6, 0)
        )

        summary_card, summary_body = self._card(
            parent,
            "结果摘要",
            "这里会展示当前寿命结果、阈值信息，以及是否启用了拟合外推。",
            subtitle_wraplength=920,
        )
        summary_card.pack(fill="x", pady=(0, 10))
        self.summary_text_holder = tk.Frame(summary_body, bg=self.colors["surface"], height=60)
        self.summary_text_holder.pack(fill="x")
        self.summary_text_holder.pack_propagate(False)
        tk.Label(
            self.summary_text_holder,
            textvariable=self.summary_var,
            bg=self.colors["surface"],
            fg=self.colors["text"],
            justify="left",
            wraplength=860,
            font=self.fonts["summary"],
        ).pack(anchor="w", fill="both")
        self.notice_banner, self.notice_label = self._banner(summary_body, self.warning_var, tone="info")
        self.notice_banner.configure(height=40)
        self.notice_banner.pack_propagate(False)
        self.notice_banner.pack(fill="x", pady=(12, 0))

        chart_card, chart_body = self._card(
            parent,
            "磨损深度演化曲线",
            "主图区会显示磨损深度随实际摆动转数的变化，并在阈值位置标出寿命结果。",
            padx=18,
            pady=18,
            subtitle_wraplength=920,
        )
        chart_card.pack(fill="both", expand=True)

        self.chart_shell = tk.Frame(chart_body, bg=self.colors["surface"], height=610)
        self.chart_shell.pack(fill="both", expand=True)
        self.chart_shell.pack_propagate(False)
        self.chart_shell.grid_columnconfigure(0, weight=1)
        self.chart_shell.grid_rowconfigure(0, weight=1)

        self.chart_holder = tk.Frame(self.chart_shell, bg=self.colors["surface"])
        self.chart_holder.grid(row=0, column=0, sticky="nsew")

        self.toolbar_holder = tk.Frame(self.chart_shell, bg=self.colors["surface"], height=42)
        self.toolbar_holder.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self.toolbar_holder.grid_propagate(False)

        self.placeholder_label = tk.Label(
            self.chart_holder,
            text="预测后，这里会显示主曲线图、阈值线和寿命标记。",
            bg=self.colors["surface_alt"],
            fg=self.colors["muted"],
            font=self.fonts["placeholder"],
            justify="center",
            padx=24,
            pady=36,
        )
        self.placeholder_label.pack(expand=True)

    def _set_notice_tone(self, tone: str) -> None:
        if tone == "warning":
            bg = self.colors["warning_soft"]
            fg = self.colors["warning"]
        elif tone == "success":
            bg = self.colors["success_soft"]
            fg = self.colors["success"]
        else:
            bg = self.colors["info_soft"]
            fg = self.colors["info"]
        self.notice_banner.configure(bg=bg)
        self.notice_label.configure(bg=bg, fg=fg)

    def load_request_into_form(self) -> None:
        request = self._read_request_from_csv_or_defaults()
        self.coating_var.set(request.get("coating_name", available_coatings()[0]))
        for field_key, _, default_value in FIELD_SPECS:
            value = request.get(field_key, default_value)
            self.entries[field_key].set(str(value))
        self.status_var.set(f"当前参数来源: {self.input_path}")
        self._set_notice_tone("info")
        self.warning_var.set("已从参数文件载入输入项。你可以直接修改后开始预测。")

    def _read_request_from_csv_or_defaults(self) -> dict:
        if self.input_path.exists():
            return load_request(self.input_path)
        return default_request()

    def collect_request(self) -> dict:
        request = {"coating_name": self.coating_var.get().strip()}
        for field_key, label_text, _ in FIELD_SPECS:
            raw_value = self.entries[field_key].get().strip()
            if raw_value == "":
                raise ValueError(f"{label_text} 不能为空。")
            request[field_key] = float(raw_value)
        return request

    def save_request_to_csv(self, request: dict) -> None:
        pd.DataFrame([request]).to_csv(self.input_path, index=False)

    def save_current_request(self) -> None:
        try:
            request = self.collect_request()
        except ValueError as exc:
            messagebox.showerror("参数错误", str(exc), parent=self.root)
            return

        self.save_request_to_csv(request)
        self.status_var.set(f"参数已保存到: {self.input_path}")
        self._set_notice_tone("success")
        self.warning_var.set("参数已保存，随时可以开始预测。")

    def run_prediction(self) -> None:
        try:
            request = self.collect_request()
            self.save_request_to_csv(request)
            normalized_request, export_df, fig, life_actual_cycles, reached, warning_text, metadata, extension_mode = run_prediction_request(request)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("预测失败", str(exc), parent=self.root)
            self._set_notice_tone("warning")
            self.warning_var.set(str(exc))
            return

        if self.latest_figure is not None:
            plt.close(self.latest_figure)

        self.latest_export_df = export_df
        self.latest_figure = fig
        self.latest_csv_path = None
        self.latest_plot_path = None
        self.save_csv_button.config(state="disabled")
        self.save_plot_button.config(state="disabled")

        self.recommendation_var.set("递推步长建议 500-2000 转。")

        mode_display = {
            "recursive": "Recursive",
            "fit_extrapolation": "Curve Fit",
            "linear_fallback": "Linear Tail",
        }.get(extension_mode, extension_mode)

        if reached and life_actual_cycles is not None:
            self.life_value_var.set(f"{life_actual_cycles:.0f}")
            self.life_sub_var.set("预测寿命总转数")
            summary_text = (
                f"当前工况在 {normalized_request['coating_name']} 涂层下的预测寿命约为 {life_actual_cycles:.0f} 转。\n"
                f"磨损阈值设定为 {normalized_request['wear_threshold_um']:.2f} um，本次结果仅显示在界面中。"
            )
        else:
            self.life_value_var.set("--")
            self.life_sub_var.set("未达到阈值")
            summary_text = (
                "在内部安全上限内未达到设定磨损阈值。\n"
                f"当前末端磨损约为 {export_df['wear_depth'].iloc[-1] * 1000.0:.3f} um。"
            )

        self.mode_value_var.set(mode_display)
        self.mode_sub_var.set(f"阈值 {normalized_request['wear_threshold_um']:.2f} um")
        self.save_value_var.set("Display")
        self.save_sub_var.set("本次不自动保存 CSV / PNG")

        self.summary_var.set(summary_text)
        if warning_text:
            self._set_notice_tone("warning")
            self.warning_var.set(warning_text)
        else:
            self._set_notice_tone("success")
            self.warning_var.set("当前阈值位于训练范围内，已使用纯递推预测。")

        self.status_var.set(
            f"参数文件: {self.input_path}\n"
            "预测结果已显示在界面中，本次未自动保存 CSV / PNG。"
        )
        if self.pending_render_job is not None:
            self.root.after_cancel(self.pending_render_job)
            self.pending_render_job = None
        self.root.update_idletasks()
        self.render_figure(fig)

    def _fit_figure_to_chart_holder(self, fig) -> None:
        if self.chart_holder is None:
            return
        holder_width = self.chart_holder.winfo_width()
        holder_height = self.chart_holder.winfo_height()
        if holder_width <= 1 or holder_height <= 1:
            holder_width = max(self.chart_shell.winfo_width() if self.chart_shell is not None else 0, 760)
            toolbar_height = self.toolbar_holder.winfo_height() if self.toolbar_holder is not None else 0
            holder_height = max((self.chart_shell.winfo_height() if self.chart_shell is not None else 0) - toolbar_height - 6, 320)
        dpi = max(float(fig.get_dpi()), 1.0)
        fig.set_size_inches(holder_width / dpi, holder_height / dpi, forward=True)

    def render_figure(self, fig) -> None:
        self.pending_render_job = None
        self.placeholder_label.pack_forget()
        if self.result_canvas is not None:
            self.result_canvas.get_tk_widget().destroy()
            self.result_canvas = None
        if self.result_toolbar is not None:
            self.result_toolbar.destroy()
            self.result_toolbar = None

        self._fit_figure_to_chart_holder(fig)
        self.result_canvas = FigureCanvasTkAgg(fig, master=self.chart_holder)
        canvas_widget = self.result_canvas.get_tk_widget()
        canvas_widget.configure(highlightthickness=0, borderwidth=0)
        canvas_widget.pack(fill="both", expand=True)
        self.root.update_idletasks()
        self._fit_figure_to_chart_holder(fig)
        self.result_canvas.draw()

        self.result_toolbar = NavigationToolbar2Tk(self.result_canvas, self.toolbar_holder, pack_toolbar=False)
        self.result_toolbar.update()
        self.result_toolbar.pack(fill="x")

    def save_csv_as(self) -> None:
        if self.latest_export_df is None:
            return
        target = filedialog.asksaveasfilename(
            parent=self.root,
            title="另存预测数据 CSV",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")],
            initialfile=self.latest_csv_path.name if self.latest_csv_path else "prediction.csv",
        )
        if not target:
            return
        self.latest_export_df.to_csv(target, index=False)
        self.status_var.set(f"已另存预测数据: {target}")
        self.save_value_var.set("Exported")
        self.save_sub_var.set("CSV 已另存")

    def save_plot_as(self) -> None:
        if self.latest_figure is None:
            return
        target = filedialog.asksaveasfilename(
            parent=self.root,
            title="另存曲线图片 PNG",
            defaultextension=".png",
            filetypes=[("PNG Files", "*.png")],
            initialfile=self.latest_plot_path.name if self.latest_plot_path else "prediction.png",
        )
        if not target:
            return
        self.latest_figure.savefig(target, dpi=300)
        self.status_var.set(f"已另存曲线图片: {target}")
        self.save_value_var.set("Exported")
        self.save_sub_var.set("PNG 已另存")

    def run(self) -> None:
        self.root.mainloop()


def run_test_mode() -> None:
    root = project_root()
    input_path = root / "app_workspace" / "prediction_input.csv"
    request = load_request(input_path)
    normalized_request, export_df, fig, life_actual_cycles, reached, warning_text, _, extension_mode = run_prediction_request(request)
    plt.close(fig)

    if reached and life_actual_cycles is not None:
        print(f"Predicted life: {life_actual_cycles:.2f} cycles")
    else:
        print("Threshold not reached before internal safety limit")
    print(f"Prediction mode: {extension_mode}")
    if warning_text:
        print(f"Warning: {warning_text}")
    print("Result displayed/tested only; CSV and PNG were not saved.")


def main() -> None:
    parser = argparse.ArgumentParser(description="GUI entry for bearing life prediction.")
    parser.add_argument("--test-mode", action="store_true", help="Run one prediction without opening the GUI.")
    args = parser.parse_args()

    if args.test_mode:
        run_test_mode()
        return

    app = PredictionGUI()
    app.run()


if __name__ == "__main__":
    main()

