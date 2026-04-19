from __future__ import annotations

import numpy as np
import torch

from device_runtime import resolve_training_device
from train_real_wear_models import Standardizer, StressNet, build_training_arrays, load_cases


def main() -> None:
    device, device_label = resolve_training_device("directml")
    if device_label != "directml":
        raise RuntimeError(f"Expected directml device, but resolved to: {device_label}")

    summary_df, case_tables = load_cases()
    if "has_measured_pressure" in summary_df.columns:
        measured_rows = summary_df[summary_df["has_measured_pressure"].astype(bool)].reset_index(drop=True)
        if not measured_rows.empty:
            case_tables = {str(row.file_name): case_tables[str(row.file_name)] for row in measured_rows.itertuples(index=False)}

    train_x, train_y = build_training_arrays(case_tables)
    if len(train_x) == 0:
        raise RuntimeError("No training samples were found for the DirectML smoke test.")

    batch_size = min(8, len(train_x))
    batch_x = np.asarray(train_x[:batch_size], dtype=np.float32)
    batch_y = np.asarray(train_y[:batch_size], dtype=np.float32)

    x_scaler = Standardizer().fit(batch_x).to_torch(device)
    y_scaler = Standardizer().fit(batch_y).to_torch(device)
    x_t = torch.tensor(x_scaler.transform_np(batch_x), dtype=torch.float32, device=device)
    y_t = torch.tensor(y_scaler.transform_np(batch_y), dtype=torch.float32, device=device)

    model = StressNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
    loss_fn = torch.nn.MSELoss()

    model.train()
    optimizer.zero_grad()
    pred = model(x_t)
    loss = loss_fn(pred, y_t)
    loss.backward()
    optimizer.step()

    print(f"device: {device_label}")
    print(f"batch_size: {batch_size}")
    print(f"loss: {float(loss.item()):.6f}")
    print("smoke_train_directml: ok")


if __name__ == "__main__":
    main()
