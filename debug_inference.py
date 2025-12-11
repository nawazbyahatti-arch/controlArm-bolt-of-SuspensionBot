import joblib
import numpy as np
from compute_features import compute_features_for_input

ann_model = joblib.load("models/ann_model.joblib")
scaler = joblib.load("models/scaler.joblib")
le = joblib.load("models/label_encoder.joblib")

def test_case(name, inputs):
    features, metrics = compute_features_for_input(inputs)
    Xs = scaler.transform(features.reshape(1,-1))
    proba = ann_model.predict_proba(Xs)[0]
    pred = le.inverse_transform([np.argmax(proba)])[0]
    print(f"\n{name}")
    print("Metrics:", metrics)
    print("Predicted:", pred, "Proba:", proba)

setA = {"diameter_mm":12,"fillet_radius_mm":1.5,"length_mm":40,
        "axial_load_N":10000,"bending_load_N":200,"bending_distance_mm":20,
        "ultimate_strength_MPa":600,"yield_strength_MPa":350,"endurance_limit_MPa":240}

setB = {"diameter_mm":12,"fillet_radius_mm":1.5,"length_mm":40,
        "axial_load_N":40000,"bending_load_N":3000,"bending_distance_mm":20,
        "ultimate_strength_MPa":600,"yield_strength_MPa":350,"endurance_limit_MPa":240}

setC = {"diameter_mm":12,"fillet_radius_mm":1.5,"length_mm":40,
        "axial_load_N":80000,"bending_load_N":5000,"bending_distance_mm":20,
        "ultimate_strength_MPa":600,"yield_strength_MPa":350,"endurance_limit_MPa":240}

test_case("Set A (Safe)", setA)
test_case("Set B (Risk)", setB)
test_case("Set C (Failure)", setC)
