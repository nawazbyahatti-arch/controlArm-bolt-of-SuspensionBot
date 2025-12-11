# app.py
# Bolt Stress & Failure Bot — physics-first decision + runtime controls (D.1)
# + human confirm (D.2) + extended logging (D.3)

import streamlit as st
import numpy as np
import joblib
import os
import io
import csv
import json
import time

# ---------------- CONFIG ----------------
ANN_MODEL_PATH = "models/ann_model.joblib"
ANN_SCALER_PATH = "models/scaler.joblib"
ANN_LABEL_ENCODER_PATH = "models/label_encoder.joblib"

AUTO_CORRECT_YIELD = True
DEFAULT_YIELD_RATIO = 0.7
FORCE_PHYSICS_OVERRIDE = True
OOD_LOAD_THRESHOLD = 1e6  # N

# ---------------- Safe loader ----------------
@st.cache_data(show_spinner=False)
def safe_joblib_load(path):
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path, mmap_mode='r')
    except TypeError:
        return joblib.load(path)
    except Exception as e:
        st.warning(f"Could not load {path}: {e}")
        return None

# ---------------- Feature function ----------------
def compute_features_for_input(inputs):
    import numpy as _np
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
    a = bending_distance_mm + eps

    area_mm2 = _np.pi * (d**2) / 4.0
    d_over_r = d / r
    fillet_ratio = r / d
    d2 = d**2
    d3 = d**3
    inv_d = 1.0 / d
    log_d = _np.log(d + 1e-6)
    sqrt_d = _np.sqrt(d)

    moment_Nmm = bending_load_N * bending_distance_mm
    I_mm4 = _np.pi * (d**4) / 64.0
    section_modulus_mm3 = I_mm4 / (d/2.0 + eps)

    # stresses in MPa (N/mm^2)
    sigma_axial_MPa = (axial_load_N / (area_mm2 + eps))
    sigma_bending_MPa = (moment_Nmm * (d/2.0)) / (I_mm4 + eps)
    sigma_max_MPa = sigma_axial_MPa + sigma_bending_MPa
    sigma_von_mises_MPa = _np.abs(sigma_max_MPa)

    safety_factor_yield = (yield_strength_MPa / (sigma_von_mises_MPa + eps))
    safety_factor_ultimate = (ultimate_strength_MPa / (sigma_von_mises_MPa + eps))

    sigma_a = _np.abs(sigma_bending_MPa)
    sigma_m = sigma_axial_MPa
    if ultimate_strength_MPa > 0:
        sigma_a_allow = endurance_limit_MPa * max(0.0, (1.0 - sigma_m / ultimate_strength_MPa))
    else:
        sigma_a_allow = 0.0
    fatigue_margin = (sigma_a_allow / (sigma_a + eps))

    # crude cycles estimate
    b = 0.12
    C = 1e9
    try:
        estimated_cycles = C * ((sigma_a + eps) ** (-b))
    except Exception:
        estimated_cycles = 0.0

    sigma_ratio = sigma_bending_MPa / (sigma_axial_MPa + eps)
    normalized_axial = sigma_axial_MPa / (ultimate_strength_MPa + eps)
    normalized_bending = sigma_bending_MPa / (ultimate_strength_MPa + eps)
    fillet_effect_scf = 1.0 + 2.0 * _np.sqrt(d_over_r + eps)
    log_moment = _np.log(moment_Nmm + 1.0)
    abs_bending_load = abs(bending_load_N)
    abs_axial_load = abs(axial_load_N)

    features = [
        diameter_mm,
        fillet_radius_mm,
        length_mm,
        axial_load_N,
        bending_load_N,
        bending_distance_mm,
        ultimate_strength_MPa,
        yield_strength_MPa,
        endurance_limit_MPa,
        area_mm2,
        d_over_r,
        fillet_ratio,
        d2,
        d3,
        inv_d,
        log_d,
        sqrt_d,
        moment_Nmm,
        I_mm4,
        section_modulus_mm3,
        sigma_bending_MPa,
        sigma_axial_MPa,
        sigma_max_MPa,
        sigma_von_mises_MPa,
        safety_factor_yield,
        safety_factor_ultimate,
        fatigue_margin,
        estimated_cycles,
        sigma_ratio,
        normalized_axial,
        normalized_bending,
        fillet_effect_scf,
        log_moment,
        abs_bending_load,
        abs_axial_load
    ]

    features = _np.array(features, dtype=float)
    metrics = {
        "area_mm2": area_mm2,
        "sigma_axial_MPa": sigma_axial_MPa,
        "sigma_bending_MPa": sigma_bending_MPa,
        "sigma_max_MPa": sigma_max_MPa,
        "sigma_von_mises_MPa": sigma_von_mises_MPa,
        "safety_factor_yield": safety_factor_yield,
        "safety_factor_ultimate": safety_factor_ultimate,
        "fatigue_margin": fatigue_margin,
        "estimated_cycles_to_failure": estimated_cycles
    }
    return features, metrics

# ---------------- UI ----------------
st.set_page_config(page_title="Bolt Stress & Failure Bot", layout="wide")
st.title("Suspension Control Arm Bolt — Stress Concentration & Failure Bot")

# ---------------- Sidebar runtime controls (D.1) ----------------
st.sidebar.markdown("## Runtime settings")
allow_mild_override = st.sidebar.checkbox("Allow ML-guided mild-failure override", value=True)
safe_thresh = st.sidebar.slider("Safe threshold (%)", min_value=0, max_value=100, value=30)
risk_lower = st.sidebar.slider("Risk lower (%)", min_value=0, max_value=100, value=30)
risk_upper = st.sidebar.slider("Risk upper (%)", min_value=0, max_value=100, value=70)
mild_sf_min = st.sidebar.slider("Mild SF min (fraction)", 0.0, 1.5, 0.8)
mild_sf_max = st.sidebar.slider("Mild SF max (fraction)", 0.0, 1.5, 1.0)
mild_fm_min = st.sidebar.slider("Mild FM min (fraction)", 0.0, 1.5, 0.8)
mild_fm_max = st.sidebar.slider("Mild FM max (fraction)", 0.0, 1.5, 1.0)
st.sidebar.markdown("---")

# status & lazy load models
st.markdown("**App status**")
st.write("Files in folder:", os.listdir('.'))

ann_model = safe_joblib_load(ANN_MODEL_PATH)
ann_scaler = safe_joblib_load(ANN_SCALER_PATH)
ann_label_encoder = safe_joblib_load(ANN_LABEL_ENCODER_PATH)

st.write("ANN model present:", ann_model is not None)
st.write("ANN scaler present:", ann_scaler is not None)
st.write("ANN label encoder present:", ann_label_encoder is not None)

# Input form
with st.form("input_form"):
    st.subheader("Geometry & Loads")
    col1, col2 = st.columns(2)
    with col1:
        diameter_mm = st.number_input("Bolt diameter (mm)", value=12.0, min_value=1.0, step=0.5)
        fillet_radius_mm = st.number_input("Fillet radius (mm)", value=1.5, min_value=0.0, step=0.1)
        length_mm = st.number_input("Loaded length (mm)", value=40.0, min_value=1.0, step=1.0)
    with col2:
        axial_load_N = st.number_input("Axial load (N) — tensile positive", value=0.0, step=10.0)
        bending_load_N = st.number_input("Bending load (N)", value=200.0, step=10.0)
        bending_distance_mm = st.number_input("Bending lever arm (mm)", value=20.0, min_value=0.0, step=1.0)

    st.subheader("Material")
    ultimate_strength_MPa = st.number_input("UTS (MPa)", value=600.0, min_value=1.0, step=1.0)
    yield_strength_MPa = st.number_input("Yield (MPa)", value=350.0, min_value=0.0, step=1.0)
    endurance_limit_MPa = st.number_input("Endurance limit (MPa) — 0 to auto", value=0.0, step=1.0)

    submitted = st.form_submit_button("Analyze & Predict")

# placeholders
ml_probs = None
ml_classes = None

if submitted:
    inputs = {
        'diameter_mm': diameter_mm,
        'fillet_radius_mm': fillet_radius_mm,
        'length_mm': length_mm,
        'axial_load_N': axial_load_N,
        'bending_load_N': bending_load_N,
        'bending_distance_mm': bending_distance_mm,
        'ultimate_strength_MPa': ultimate_strength_MPa,
        'yield_strength_MPa': yield_strength_MPa,
        'endurance_limit_MPa': None if endurance_limit_MPa <= 0 else endurance_limit_MPa
    }

    # validation & auto-correct
    warnings = []
    if ultimate_strength_MPa <= 0:
        warnings.append("UTS must be > 0 MPa.")
    if yield_strength_MPa <= 0:
        warnings.append("Yield strength must be > 0 MPa.")
    if yield_strength_MPa > ultimate_strength_MPa:
        warnings.append(f"Yield ({yield_strength_MPa} MPa) > UTS ({ultimate_strength_MPa} MPa).")
        if AUTO_CORRECT_YIELD:
            corrected = round(DEFAULT_YIELD_RATIO * ultimate_strength_MPa, 2)
            warnings.append(f"Auto-correcting Yield to {corrected} MPa (={DEFAULT_YIELD_RATIO} * UTS).")
            yield_strength_MPa = corrected
            inputs['yield_strength_MPa'] = yield_strength_MPa

    if diameter_mm <= 0 or diameter_mm > 200:
        warnings.append("Bolt diameter looks unrealistic. Enter diameter in mm (e.g., 6–30).")
    if bending_distance_mm > 10000:
        warnings.append("Bending lever arm unusually large (>10,000 mm). Check units.")
    if abs(bending_load_N) > 1e6:
        warnings.append("Bending load extremely large (>1e6 N). Check units.")
    if abs(axial_load_N) > 1e6:
        warnings.append("Axial load extremely large (>1e6 N). Check units.")

    if warnings:
        for w in warnings:
            st.warning(w)

    # compute and show metrics
    features, metrics = compute_features_for_input(inputs)
    st.subheader("Derived metrics")
    st.write(metrics)

    # physics checks
    SF_yield = metrics.get("safety_factor_yield", float('inf'))
    fatigue_margin = metrics.get("fatigue_margin", float('inf'))
    sigma_max = metrics.get("sigma_max_MPa", 0.0)

    physics_outcome = "inconclusive"
    physics_reason = ""

    axial_N = inputs.get("axial_load_N", 0.0)
    bending_N = inputs.get("bending_load_N", 0.0)
    if abs(axial_N) > OOD_LOAD_THRESHOLD or abs(bending_N) > OOD_LOAD_THRESHOLD:
        physics_outcome = "failure"
        physics_reason = f"Loads exceed {OOD_LOAD_THRESHOLD} N — out of realistic range."

    if physics_outcome != "failure":
        if SF_yield < 1.0:
            physics_outcome = "failure"
            physics_reason = f"Safety factor (yield) < 1.0 ({SF_yield:.2f}) → plastic collapse expected."
        elif fatigue_margin < 1.0:
            physics_outcome = "failure"
            physics_reason = f"Fatigue margin < 1.0 ({fatigue_margin:.2f}) → fatigue failure likely."
        elif inputs.get("ultimate_strength_MPa", None) is not None:
            try:
                if sigma_max >= float(inputs["ultimate_strength_MPa"]):
                    physics_outcome = "failure"
                    physics_reason = f"Combined stress ({sigma_max:.2f} MPa) ≥ UTS ({inputs['ultimate_strength_MPa']} MPa)."
            except Exception:
                pass

    if physics_outcome != "failure":
        if SF_yield >= 1.0 and fatigue_margin >= 1.0 and sigma_max < inputs.get("ultimate_strength_MPa", 1e12):
            physics_outcome = "safe"
            physics_reason = f"SF={SF_yield:.2f} >=1 and FM={fatigue_margin:.2f} >=1 and σ_max<{inputs.get('ultimate_strength_MPa')} MPa."

    # ANN inference (if available)
    failure_prob = None
    if ann_model is not None and ann_scaler is not None and ann_label_encoder is not None:
        try:
            X = features.reshape(1, -1)
            try:
                Xs = ann_scaler.transform(X)
            except Exception:
                Xs = X
            proba = ann_model.predict_proba(Xs)[0]
            classes = list(getattr(ann_label_encoder, "classes_", []))
            ml_probs = {classes[i]: float(proba[i]) for i in range(len(classes))}
            ml_classes = classes
            # robustly find failure class index
            failure_index = None
            for i, cname in enumerate(classes):
                try:
                    if str(cname).strip().lower() == "failure":
                        failure_index = i
                        break
                except Exception:
                    pass
            if failure_index is None:
                for i, cname in enumerate(classes):
                    try:
                        if "fail" in str(cname).lower():
                            failure_index = i
                            break
                    except Exception:
                        pass
            if failure_index is None:
                failure_index = len(proba) - 1
            failure_prob = float(proba[failure_index])
        except Exception as e:
            st.warning(f"ANN inference failed: {e}")
            ml_probs = None
            failure_prob = None

    # Decision logic (physics-first) with runtime mild-override control
    final_label = None
    final_reason = None

    if FORCE_PHYSICS_OVERRIDE and physics_outcome == "failure":
        # mild detection uses sidebar sliders (D.1)
        mild_sf = (SF_yield >= float(mild_sf_min) and SF_yield < float(mild_sf_max))
        mild_fm = (fatigue_margin >= float(mild_fm_min) and fatigue_margin < float(mild_fm_max))

        if allow_mild_override and (mild_sf or mild_fm) and (failure_prob is not None) and (ml_probs is not None):
            pct = failure_prob * 100.0
            # use runtime risk bounds
            if float(risk_lower) <= pct < float(risk_upper):
                final_label = "Risk"
                final_reason = (f"Mild physics failure (SF={SF_yield:.2f}, FM={fatigue_margin:.2f}) "
                                f"but ML failure_prob={pct:.2f}% -> show Risk (override enabled).")
                st.warning(f"PHYSICS: mild failure but ML suggests Risk — {final_reason}")
                rows = [(k, round(v*100,2)) for k,v in ml_probs.items()]
                st.write("ML probabilities (advisory):")
                st.table(rows)
            else:
                final_label = "Failure"
                final_reason = f"Physics -> {physics_reason}"
                st.error(f"DETERMINISTIC OVERRIDE: {final_label} — {physics_reason}")
                st.write("Probabilities (overridden):")
                st.table([("Failure", "100.00%"), ("Safe/Risk (others)", "0.00%")])
        else:
            final_label = "Failure"
            final_reason = f"Physics -> {physics_reason}"
            st.error(f"DETERMINISTIC OVERRIDE: {final_label} — {physics_reason}")
            st.write("Probabilities (overridden):")
            st.table([("Failure", "100.00%"), ("Safe/Risk (others)", "0.00%")])
    else:
        if physics_outcome == "safe":
            final_label = "Safe"
            final_reason = f"Physics -> {physics_reason}"
            st.success(f"PHYSICS: {final_label} — {physics_reason}")
            if ml_probs:
                st.info("ML advisory probabilities (physics-safe, advisory only):")
                rows = [(k, round(v*100,2)) for k,v in ml_probs.items()]
                st.table(rows)
        else:
            # physics inconclusive -> use ML, thresholds are driven by sidebar
            if failure_prob is None:
                if SF_yield >= 2.0 and fatigue_margin >= 2.0:
                    final_label = "Safe"
                    final_reason = "Fallback thresholds (SF>=2 and FM>=2)"
                elif SF_yield >= 1.0 and fatigue_margin >= 1.0:
                    final_label = "Risk"
                    final_reason = "Fallback thresholds (SF>=1 and FM>=1)"
                else:
                    final_label = "Failure"
                    final_reason = "Fallback thresholds indicate failure"
            else:
                pct = failure_prob * 100.0
                if pct < float(safe_thresh):
                    final_label = "Safe"
                    final_reason = f"ML failure prob {pct:.2f}% < {safe_thresh}% -> Safe"
                elif float(risk_lower) <= pct < float(risk_upper):
                    final_label = "Risk"
                    final_reason = f"ML failure prob {pct:.2f}% between {risk_lower}-{risk_upper}% -> Risk"
                else:
                    final_label = "Failure"
                    final_reason = f"ML failure prob {pct:.2f}% >= {risk_upper}% -> Failure"

                if ml_probs:
                    st.write("ML probabilities:")
                    rows = [(k, round(v*100,2)) for k,v in ml_probs.items()]
                    st.table(rows)

    # Ensure final_label not None
    if final_label is None:
        final_label = "Risk"
        final_reason = final_reason or "Conservative default"

    # Present final result
    st.subheader("Final decision")
    if final_label == "Safe":
        st.success(f"Final: {final_label} — {final_reason}")
    elif final_label == "Risk":
        st.warning(f"Final: {final_label} — {final_reason}")
    else:
        st.error(f"Final: {final_label} — {final_reason}")

    # Confirm / label buttons (quick human labelling) — D.2
    st.markdown("### Confirm / Label this run")
    c1, c2, c3 = st.columns(3)

    # Ensure confirmed_labels.csv exists and has header
    CONF_FILE = "confirmed_labels.csv"
    if not os.path.exists(CONF_FILE):
        try:
            with open(CONF_FILE, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "inputs", "metrics", "confirmed_label", "app_label"])
        except Exception:
            pass

    if c1.button("Confirm: SAFE"):
        with open(CONF_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([time.time(), json.dumps(inputs), json.dumps(metrics), "Safe", final_label])
        st.success("Recorded: Safe")
    if c2.button("Confirm: RISK"):
        with open(CONF_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([time.time(), json.dumps(inputs), json.dumps(metrics), "Risk", final_label])
        st.success("Recorded: Risk")
    if c3.button("Confirm: FAILURE"):
        with open(CONF_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([time.time(), json.dumps(inputs), json.dumps(metrics), "Failure", final_label])
        st.success("Recorded: Failure")

    # Video selection & display — choose by final_label or ML failure_prob (sidebar thresholds already used above)
    safe_vid = "safe.mp4"
    risk_vid = "risk.mp4"
    fail_vid = "failure.mp4"
    chosen_video = None
    verbose_reason = ""

    if final_label == "Failure":
        chosen_video = fail_vid
        verbose_reason = "Final label Failure -> failure video"
    else:
        if failure_prob is not None:
            pct = failure_prob * 100.0
            if pct < float(safe_thresh):
                chosen_video = safe_vid
                verbose_reason = f"ML failure_prob {pct:.2f}% -> SAFE video"
            elif float(risk_lower) <= pct < float(risk_upper):
                chosen_video = risk_vid
                verbose_reason = f"ML failure_prob {pct:.2f}% -> RISK video"
            else:
                chosen_video = fail_vid
                verbose_reason = f"ML failure_prob {pct:.2f}% -> FAILURE video"
        else:
            if final_label == "Safe":
                chosen_video = safe_vid
                verbose_reason = "Final label Safe -> safe video"
            elif final_label == "Risk":
                chosen_video = risk_vid
                verbose_reason = "Final label Risk -> risk video"
            else:
                chosen_video = fail_vid
                verbose_reason = "Final label Failure -> failure video"

    st.subheader("Stress Video Visualization")
    if chosen_video == safe_vid:
        st.success(f"Video status: SAFE — {verbose_reason}")
    elif chosen_video == risk_vid:
        st.warning(f"Video status: RISK — {verbose_reason}")
    else:
        st.error(f"Video status: FAILURE — {verbose_reason}")

    # Display chosen video (st.video accepts path)
    try:
        if chosen_video is not None and os.path.exists(chosen_video):
            st.video(chosen_video)
            try:
                with open(chosen_video, "rb") as vf2:
                    st.download_button("Download selected video", data=vf2.read(), file_name=os.path.basename(chosen_video), mime="video/mp4")
            except Exception:
                pass
        else:
            st.info("No matching video file found. Place safe.mp4, risk.mp4, failure.mp4 in the app folder to enable video playback.")
    except Exception as ex:
        st.error(f"Could not load video: {ex}")

    # Extended logging (D.3) - include runtime settings and ML info
    try:
        LOGFILE = "runs_log.csv"
        if not os.path.exists(LOGFILE):
            with open(LOGFILE, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp","inputs","metrics","physics_outcome","physics_reason",
                    "failure_prob","ml_probs","final_label","allow_mild_override",
                    "safe_thresh","risk_lower","risk_upper","mild_sf_min","mild_sf_max",
                    "mild_fm_min","mild_fm_max"
                ])
        with open(LOGFILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                time.time(),
                json.dumps(inputs),
                json.dumps(metrics),
                physics_outcome,
                physics_reason,
                None if failure_prob is None else round(float(failure_prob), 6),
                json.dumps(ml_probs),
                final_label,
                bool(allow_mild_override),
                safe_thresh, risk_lower, risk_upper, mild_sf_min, mild_sf_max, mild_fm_min, mild_fm_max
            ])
    except Exception as e:
        # non-fatal
        st.write("Logging failed:", e)

st.markdown("---")
st.write("If something still looks off, copy the Derived metrics and model probabilities and paste them here and I'll help interpret.")












