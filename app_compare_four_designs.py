# app_compare_four_designs.py
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Compare 4 Bicycle Frame Designs", layout="wide")
st.title("Compare 4 Bicycle Frame Designs — Pick the best for a given load")

st.markdown("""
**How to use**
- Choose **CSV mode** if you have ANSYS results exported to CSV with columns: `Design`, `Load_text` (or `Primary_load`), `Stress_MPa`, `Deformation_mm`.
- Or choose **Manual mode** to type stress/deformation or geometry for each of the 4 designs.
""")

mode = st.radio("Select input mode", ["CSV mode (ANSYS results)", "Manual mode (enter 4 designs)"])

################################################################################
# Helper functions
################################################################################
def score_table(df, stress_col="Stress_MPa", deform_col="Deformation_mm", stress_w=0.5, deform_w=0.5):
    # For the selected rows (same load), compute normalized score and best
    s_min, s_max = df[stress_col].min(), df[stress_col].max()
    d_min, d_max = df[deform_col].min(), df[deform_col].max()
    # handle flat ranges
    if s_max == s_min:
        df["stress_norm"] = 0.0
    else:
        df["stress_norm"] = (df[stress_col] - s_min) / (s_max - s_min)
    if d_max == d_min:
        df["def_norm"] = 0.0
    else:
        df["def_norm"] = (df[deform_col] - d_min) / (d_max - d_min)
    # score (lower better). Weighted sum
    df["score"] = df["stress_norm"] * stress_w + df["def_norm"] * deform_w
    df_sorted = df.sort_values("score").reset_index(drop=True)
    return df_sorted

################################################################################
# CSV MODE - expects a CSV with results for designs across loads
################################################################################
if mode == "CSV mode (ANSYS results)":
    st.header("CSV mode — upload ANSYS results")
    st.markdown("""
    The CSV should contain rows for each (Design × Load) pair, e.g.:
    `Design`, `Load_text`, `Primary_load`, `Stress_MPa`, `Deformation_mm`
    Or at minimum: `Design`, `Load_text`, `Stress_MPa`, `Deformation_mm`.
    """)
    uploaded = st.file_uploader("Upload ANSYS results CSV", type=["csv"])
    stress_weight = st.slider("Stress weight (relative importance)", 0.0, 1.0, 0.5)
    if uploaded is not None:
        df_all = pd.read_csv(uploaded)
        st.write("Preview of uploaded file:")
        st.dataframe(df_all.head(50))
        # Normalize column names (common variants)
        cols = {c.lower(): c for c in df_all.columns}
        # try to find columns for design, load, stress, deformation
        # case-insensitive match
        def find_col(possible_names):
            for n in possible_names:
                if n.lower() in cols:
                    return cols[n.lower()]
            return None
        col_design = find_col(["Design", "design", "Frame", "frame", "DesignName"])
        col_load_text = find_col(["Load_text", "Load", "load_text", "LoadText", "Primary_load", "primary_load"])
        col_stress = find_col(["Stress_MPa", "Stress (MPa)", "stress_mpa", "Stress"])
        col_def = find_col(["Deformation_mm", "Deformation (mm)", "deformation_mm", "Deformation"])
        if not col_design or not col_stress or not col_def or not col_load_text:
            st.error("Could not find required columns. Required: Design, Load_text (or Primary_load), Stress_MPa, Deformation_mm. Check column names.")
        else:
            st.success(f"Using columns: Design='{col_design}', Load='{col_load_text}', Stress='{col_stress}', Deformation='{col_def}'")
            # Ask user for the load to compare
            load_input = st.text_input("Enter the exact Load_text to analyze (example: '1500 and 270') or enter numeric primary load (e.g. 1500):")
            if st.button("Compare designs for this load (CSV)"):
                if load_input.strip() == "":
                    st.error("Please enter a load string (exact match) or numeric load.")
                else:
                    # try numeric parse
                    try:
                        primary_load_val = int(''.join(ch for ch in load_input if ch.isdigit()))
                    except Exception:
                        primary_load_val = None
                    # filter rows where col_load_text equals input or numeric match to digits
                    df_all[col_load_text] = df_all[col_load_text].astype(str)
                    subset = df_all[df_all[col_load_text] == load_input]
                    if subset.empty and primary_load_val is not None:
                        # try matching digits inside load string
                        subset = df_all[df_all[col_load_text].str.contains(str(primary_load_val), na=False)]
                    if subset.empty:
                        st.error("No rows found for that load. Check exact string or upload data where load strings match.")
                    else:
                        # We expect 4 designs — but proceed with whatever designs exist
                        designs_present = subset[col_design].unique().tolist()
                        st.write(f"Found designs for this load: {designs_present}")
                        # prepare numeric columns
                        subset[col_stress] = pd.to_numeric(subset[col_stress], errors="coerce")
                        subset[col_def] = pd.to_numeric(subset[col_def], errors="coerce")
                        subset = subset.dropna(subset=[col_stress, col_def])
                        if subset.empty:
                            st.error("Stress/Deformation values are missing or non-numeric for this load.")
                        else:
                            sorted_df = score_table(subset.rename(columns={col_design:"Design", col_stress:"Stress_MPa", col_def:"Deformation_mm"}),
                                                    "Stress_MPa","Deformation_mm", stress_w=stress_weight, deform_w=1-stress_weight)
                            st.subheader("Comparison table (lower score is better)")
                            st.dataframe(sorted_df[["Design","Stress_MPa","Deformation_mm","score"]].reset_index(drop=True))
                            best = sorted_df.iloc[0]
                            st.success(f"Recommended design: **{best['Design']}** (score {best['score']:.4f})")
                            st.markdown("### Bar chart of scores")
                            chart_df = sorted_df[["Design","score"]].set_index("Design")
                            st.bar_chart(chart_df)
                            st.markdown("### Quick engineering suggestions")
                            st.write(" - Lower stress: increase tube section (increase diameter or thickness) or change material; reduce applied load.")
                            st.write(" - Lower deformation: increase section modulus (moment of inertia) or material stiffness (E).")
                            st.write(" - If safety/fatigue data available, prefer designs with higher safety factor and predicted cycles.")

################################################################################
# MANUAL MODE - user enters 4 designs stress & deformation or geometry
################################################################################
else:
    st.header("Manual mode — enter values for 4 designs")
    st.markdown("You can either input **Stress (MPa) & Deformation (mm)** directly for each design, or press the 'Compute from geometry' checkbox and provide geometry + loads per design.")
    stress_weight = st.slider("Stress weight (relative importance)", 0.0, 1.0, 0.5)
    use_geometry = st.checkbox("Compute stress/deformation from geometry (instead of entering numbers manually)?", value=False)
    designs = []
    cols = st.columns(4)
    for i in range(4):
        with cols[i]:
            st.markdown(f"### Design {i+1}")
            name = st.text_input(f"Name (Design {i+1})", value=f"Design_{i+1}", key=f"name{i}")
            if not use_geometry:
                st.number_input(f"Stress (MPa) - {name}", min_value=0.0, value=100.0, key=f"stress{i}")
                st.number_input(f"Deformation (mm) - {name}", min_value=0.0, value=0.1, key=f"def{i}")
            else:
                # geometry inputs (simple hollow circular tube)
                d_o = st.number_input(f"Outer dia (mm) - {name}", min_value=10.0, value=30.0, key=f"do{i}")
                t = st.number_input(f"Thickness (mm) - {name}", min_value=0.2, value=2.0, key=f"th{i}")
                L = st.number_input(f"Lever length (mm) - {name}", min_value=50.0, value=350.0, key=f"L{i}")
                # for geometry mode we will also need loads; ask single global load below or per design
                st.write("Note: loads will be taken from the 'Load inputs' panel below.")
            designs.append(name)

    st.markdown("### Load inputs (applied to all designs for fair comparison)")
    load_text = st.text_input("Load description (example: '1500 and 270')", value="USER_LOAD")
    rider_mass = st.number_input("Rider mass (kg)", min_value=10.0, value=75.0)
    axial_fraction = st.slider("Axial force fraction of weight (0–2×g)", 0.0, 2.0, 0.3)
    trans_fraction = st.slider("Transverse force fraction of weight (0–2×g)", 0.0, 2.0, 0.6)
    g=9.81
    axial_F = rider_mass * g * axial_fraction
    trans_F = rider_mass * g * trans_fraction

    if st.button("Compare 4 designs (manual)"):
        rows = []
        for i in range(4):
            name = st.session_state.get(f"name{i}", f"Design_{i+1}")
            if not use_geometry:
                stress = st.session_state.get(f"stress{i}")
                deform = st.session_state.get(f"def{i}")
                rows.append({"Design": name, "Load_text": load_text, "Stress_MPa": float(stress), "Deformation_mm": float(deform)})
            else:
                d_o = st.session_state.get(f"do{i}")
                t = st.session_state.get(f"th{i}")
                L = st.session_state.get(f"L{i}")
                # compute approximate stresses using simple beam/tube formulas
                # area and I
                di = d_o - 2*t
                if di <= 0:
                    st.error(f"Design {name}: invalid geometry (inner dia <= 0). Increase outer diameter or reduce thickness.")
                    continue
                A = np.pi*(d_o**2 - di**2)/4.0
                I = np.pi*(d_o**4 - di**4)/64.0
                sigma_axial = axial_F / A
                M = trans_F * L
                c = d_o/2.0
                sigma_bend = M * c / I
                sigma_vm = np.sqrt(sigma_axial**2 + 3*(sigma_bend**2))
                # deflection approximation
                E = 210000.0
                delta = trans_F * (L**3) / (3.0 * E * I)
                rows.append({"Design": name, "Load_text": load_text, "Stress_MPa": float(sigma_vm), "Deformation_mm": float(delta)})

        df_manual = pd.DataFrame(rows)
        st.subheader("Raw computed / input values for the 4 designs")
        st.dataframe(df_manual)
        df_scored = score_table(df_manual, "Stress_MPa", "Deformation_mm", stress_w=stress_weight, deform_w=1-stress_weight)
        st.subheader("Comparison (lower score is better)")
        st.dataframe(df_scored[["Design","Stress_MPa","Deformation_mm","score"]].reset_index(drop=True))
        best = df_scored.iloc[0]
        st.success(f"Recommended design: **{best['Design']}** (score {best['score']:.4f})")
        st.markdown("### Chart")
        st.bar_chart(df_scored.set_index("Design")[["score"]])
        st.markdown("### Suggestions")
        st.write("- If the chosen design has high stress: increase thickness/diameter or change material.")
        st.write("- If deformation is high: increase moment of inertia (bigger section) or use stiffer material (higher E).")
