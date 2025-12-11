import numpy as np

def compute_features_for_input(inputs):
    diameter_mm = float(inputs.get('diameter_mm', 0.0))
    fillet_radius_mm = float(inputs.get('fillet_radius_mm', 0.0))
    length_mm = float(inputs.get('length_mm', 0.0))
    axial_load_N = float(inputs.get('axial_load_N', 0.0))
    bending_load_N = float(inputs.get('bending_load_N', 0.0))
    bending_distance_mm = float(inputs.get('bending_distance_mm', 0.0))
    ultimate_strength_MPa = float(inputs.get('ultimate_strength_MPa', 0.0))
    yield_strength_MPa = float(inputs.get('yield_strength_MPa', 0.0))
    endurance_limit_MPa = inputs.get('endurance_limit_MPa', None)

    if endurance_limit_MPa is None or endurance_limit_MPa == 0:
        endurance_limit_MPa = 0.4 * ultimate_strength_MPa

    eps = 1e-9
    d = diameter_mm + eps
    r = fillet_radius_mm + eps

    area_mm2 = np.pi * (d**2) / 4.0
    moment_Nmm = bending_load_N * bending_distance_mm
    I_mm4 = np.pi * (d**4) / 64.0

    sigma_axial_MPa = (axial_load_N / (area_mm2 + eps))
    sigma_bending_MPa = (moment_Nmm * (d/2.0)) / (I_mm4 + eps)
    sigma_max_MPa = sigma_axial_MPa + sigma_bending_MPa
    sigma_von_mises_MPa = np.abs(sigma_max_MPa)

    safety_factor_yield = (yield_strength_MPa / (sigma_von_mises_MPa + eps))
    safety_factor_ultimate = (ultimate_strength_MPa / (sigma_von_mises_MPa + eps))

    sigma_a = np.abs(sigma_bending_MPa)
    sigma_m = sigma_axial_MPa
    sigma_a_allow = endurance_limit_MPa * max(0.0, (1.0 - sigma_m / ultimate_strength_MPa))
    fatigue_margin = sigma_a_allow / (sigma_a + eps)

    metrics = {
        "area_mm2": area_mm2,
        "sigma_axial_MPa": sigma_axial_MPa,
        "sigma_bending_MPa": sigma_bending_MPa,
        "sigma_max_MPa": sigma_max_MPa,
        "sigma_von_mises_MPa": sigma_von_mises_MPa,
        "safety_factor_yield": safety_factor_yield,
        "safety_factor_ultimate": safety_factor_ultimate,
        "fatigue_margin": fatigue_margin,
    }

    features = np.array([
        diameter_mm, fillet_radius_mm, length_mm, axial_load_N,
        bending_load_N, bending_distance_mm, ultimate_strength_MPa,
        yield_strength_MPa, endurance_limit_MPa, area_mm2,
        sigma_axial_MPa, sigma_bending_MPa, sigma_max_MPa,
        sigma_von_mises_MPa, safety_factor_yield, safety_factor_ultimate,
        fatigue_margin
    ], dtype=float)

    return features, metrics
