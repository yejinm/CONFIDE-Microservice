import json
import numpy as np
from pathlib import Path

# Target system
SYSTEM = "acmeair"
ROOT = Path(__file__).resolve().parents[3]

# 1) Load class_order
order_path = ROOT / "data" / "processed" / "fusion" / f"{SYSTEM}_class_order.json"
with open(order_path, 'r') as f:
    class_order = json.load(f)
class_set = set(class_order)

# 2) Load generated S_temp
temp_path = ROOT / "data" / "processed" / "temporal" / f"{SYSTEM}_S_temp.npy"
S_temp = np.load(temp_path)

# 3) Analyze results
active_indices = np.where(np.diag(S_temp) > 0)[0]
print(f"--- System: {SYSTEM} ---")
print(f"Total classes in order: {len(class_order)}")
print(f"Classes with temporal data: {len(active_indices)}")

print("\n[Matched Classes]")
for idx in active_indices:
    print(f" - {class_order[idx]}")

if len(active_indices) < 5:
    print("\n[WARNING] Very few matches! Check if your ENDPOINT_MAPS in build_S_temp.py matches these names exactly.")
