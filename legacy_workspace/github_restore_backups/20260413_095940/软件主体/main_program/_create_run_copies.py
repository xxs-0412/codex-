import shutil
from pathlib import Path

data_dir = Path(r"c:\Users\28382\Desktop\毕业设计\Bearing training\磨损数据（改）")
mapping = {
    "run1.csv": "试验1.csv",
    "run3.csv": "试验3.csv",
    "run4.csv": "试验4.csv",
    "run5.csv": "试验5.csv",
    "run6.csv": "试验6.csv",
    "run7.csv": "试验7.csv",
    "run8.csv": "试验8.csv",
    "run9.csv": "试验9.csv",
    "run10.csv": "试验10.csv",
    "run11.csv": "试验11.csv",
    "run12.csv": "试验12.csv",
    "run13.csv": "试验13.csv",
    "run14.csv": "试验14.csv",
}

for run_name, test_name in mapping.items():
    src = data_dir / test_name
    dst = data_dir / run_name
    if dst.exists():
        print(f"{run_name} already exists, skip")
    elif src.exists():
        shutil.copy2(src, dst)
        print(f"{test_name} -> {run_name}")
    else:
        print(f"{test_name} not found!")
