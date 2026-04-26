from __future__ import annotations

import json
from pathlib import Path

import common_fixed_split as common


ROOT_DIR = Path(__file__).resolve().parents[2]
TRANSFORMER_CONFIG_PATH = ROOT_DIR / "结果" / "17测试集" / "02_transformer升级" / "最终配置.json"


def load_best_transformer_config() -> common.TransformerConfig:
    payload = json.loads(TRANSFORMER_CONFIG_PATH.read_text(encoding="utf-8"))
    return common.TransformerConfig(
        seq_len=int(payload["seq_len"]),
        d_model=int(payload["d_model"]),
        nhead=int(payload["nhead"]),
        num_layers=int(payload["num_layers"]),
        dim_ff=int(payload["dim_ff"]),
        dropout=float(payload["dropout"]),
        epochs=int(payload["epochs"]),
        learning_rate=float(payload["learning_rate"]),
        weight_decay=float(payload["weight_decay"]),
    )


def lstm_baseline_config() -> common.TransformerConfig:
    return common.TransformerConfig(
        seq_len=6,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_ff=64,
        dropout=0.0,
        epochs=1200,
        learning_rate=1e-3,
        weight_decay=1e-6,
    )
