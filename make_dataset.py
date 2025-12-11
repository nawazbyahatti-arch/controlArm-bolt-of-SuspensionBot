# make_dataset.py - balanced 3-class dataset with clearer Risk definition
import pandas as pd
import numpy as np
from compute_features import compute_features_for_input

rows = []

# finer sweep for richer dataset
axial_loads = np.linspace(1000, 80000, 80)   # 80 axial points
bending_loads = np.linspace(100, 5000, 80)   # 80 bending points

for axial in axial_loads:
    for bending in bending_loads:
        inputs = {
            "diameter_mm": 12,
            "fillet_radius_mm": 1.5,
            "length_mm": 40,
            "axial_load_N": float(axial),
            "bending_load_N": float(bending),
            "bending_distance_mm": 20,
            "ultimate_strength_MPa": 600,
            "yield_strength_MPa": 350,
            "endurance_limit_MPa": 240,
        }
        features, metrics = compute_features_for_input(inputs)

        SF = float(metrics["safety_factor_yield"])
        FM = float(metrics["fatigue_margin"])
        sigma = float(metrics["sigma_max_MPa"])
        UTS = inputs["ultimate_strength_MPa"]

        # Labeling logic:
        # - Failure: definite failure by physics
        # - Risk: borderline (SF and FM between 1 and 2, and stress below UTS)
        # - Safe: otherwise (SF>=2 and FM>=2 and stress < UTS)
        label = "Safe"
        if SF < 1.0 or FM < 1.0 or sigma >= UTS:
            label = "Failure"
        elif (1.0 <= SF < 2.0) or (1.0 <= FM < 2.0):
            # only mark Risk if stress still below UTS
            if sigma < UTS:
                label = "Risk"

        row = dict(zip([f"f{i}" for i in range(len(features))], features))
        row["label"] = label
        rows.append(row)

df = pd.DataFrame(rows)

# If any class is empty, bail out
counts = df['label'].value_counts()
print("Raw class counts:", counts.to_dict())

# Balance dataset: sample equal per class based on the smallest class size
safe_df = df[df["label"] == "Safe"]
risk_df = df[df["label"] == "Risk"]
fail_df = df[df["label"] == "Failure"]

min_len = min(len(safe_df), len(risk_df), len(fail_df))
if min_len == 0:
    raise RuntimeError("One of the classes has zero examples. Adjust ranges or labelling thresholds.")

df_balanced = pd.concat([
    safe_df.sample(min_len, random_state=42),
    risk_df.sample(min_len, random_state=42),
    fail_df.sample(min_len, random_state=42)
])

df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

df_balanced.to_csv("bolt_dataset.csv", index=False)
print("Balanced dataset with 3 classes saved as bolt_dataset.csv")
print("Total samples:", len(df_balanced))
print("Safe:", (df_balanced['label'] == 'Safe').sum())
print("Risk:", (df_balanced['label'] == 'Risk').sum())
print("Failure:", (df_balanced['label'] == 'Failure').sum())


