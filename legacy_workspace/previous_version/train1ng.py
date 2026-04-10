import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# 1. 路径自动定位与数据加载
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(current_dir, "bearing_dataset")
search_path = os.path.join(data_folder, "*.csv")

all_files = glob.glob(search_path)
if not all_files:
    print(f"❌ 错误：未在 {data_folder} 下找到数据！")
    exit()

df_list = [pd.read_csv(f) for f in all_files]
full_data = pd.concat(df_list, ignore_index=True)

# 【核心特征】：彻底剔除 stress，只保留 4 个独立自变量
features = ['cycle', 'F2_actual', 'd1', 'Cr']
target = ['wear_depth']

scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
X_raw = scaler_x.fit_transform(full_data[features])
Y_raw = scaler_y.fit_transform(full_data[target])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_tensor = torch.tensor(X_raw.astype(np.float32)).to(device)
Y_tensor = torch.tensor(Y_raw.astype(np.float32)).to(device)

# ==========================================
# 2. 建立神经网络 (针对 RTX 3060 优化的深度网络)
# ==========================================
model = nn.Sequential(
    nn.Linear(len(features), 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print(f"🚀 开始 10000 轮深度学习 (输入特征: {features})...")

for epoch in range(10001):
    loss = nn.MSELoss()(model(X_tensor), Y_tensor)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    if epoch % 2000 == 0:
        print(f"进度: {epoch}/10000 | 归一化误差 Loss: {loss.item():.8f}")

# ==========================================
# 3. 智能寿命判定 (用户只需输入：F2, d1, Cr)
# ==========================================
def predict_bearing_life(f2_input, d1_input, cr_input, threshold=0.005):
    # 构建预测用的时间轴 (0 到 30万转)
    future_cycles = np.linspace(0, 300000, 500)
    
    # 构造输入 DataFrame，此时不需要 stress
    predict_data = pd.DataFrame({
        'cycle': future_cycles,
        'F2_actual': f2_input,
        'd1': d1_input,
        'Cr': cr_input
    })
    
    model.eval()
    with torch.no_grad():
        x_in = torch.tensor(scaler_x.transform(predict_data).astype(np.float32)).to(device)
        preds = scaler_y.inverse_transform(model(x_in).cpu().numpy()).flatten()

    fail_idx = np.where(preds >= threshold)[0]
    
    plt.figure(figsize=(10, 6))
    plt.plot(future_cycles, preds, label='AI Wear Predict', color='blue', linewidth=2)
    plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold}mm)')
    plt.ylim(0, threshold * 1.2) # 保持阈值线在图上方
    
    if len(fail_idx) > 0:
        life = int(future_cycles[fail_idx[0]])
        plt.scatter(life, threshold, color='red', s=100, zorder=5)
        plt.annotate(f'Life: {life} Cycles', (life, threshold), xytext=(10, -20), 
                     textcoords='offset points', color='red', fontweight='bold',
                     arrowprops=dict(arrowstyle="->", color='red'))
        print(f"🎯 预测成功：工况 [Load={f2_input}N, d1={d1_input}mm, Cr={cr_input}mm] 下，寿命为 {life} 转。")
    
    plt.title(f"Bearing Life System (Model: GE{int(d1_input)}C)")
    plt.ylabel('Wear Depth (mm)'); plt.xlabel('Cycles')
    plt.legend(); plt.grid(True, alpha=0.3); plt.show()

# ==========================================
# 4. 一键预测示例 (只输入原因，不输入中间变量)
# ==========================================
# 比如你想看 GE17C 在 4000N 载荷、0.01mm 间隙下的寿命
predict_bearing_life(f2_input=10000, d1_input=4.0, cr_input=0.008)