"""
IoT Water Quality Safety Classifier (Project ML_17)
Streamlit Interactive Application

This application loads a pre-trained scikit-learn pipeline (k-NN) to classify water potability
based on physical, chemical, and simulated IoT sensor readings.

Run locally:
    streamlit run app.py
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="IoT Water Quality Classifier (ML_17)",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Dynamic Model Loading
# ---------------------------------------------------------
@st.cache_resource
def load_trained_pipeline():
    """Locate and load the pre-trained model pipeline artifact."""
    candidates = [
        Path("models") / "water_quality_pipeline.joblib",
        Path("water-quality-classifier/models") / "water_quality_pipeline.joblib",
        Path("../models") / "water_quality_pipeline.joblib",
        Path.cwd() / "models" / "water_quality_pipeline.joblib"
    ]
    for p in candidates:
        if p.is_file():
            try:
                artifact = joblib.load(p)
                return artifact, p.resolve()
            except Exception as e:
                st.error(f"Error loading pipeline from {p}: {e}")
                return None, None
    return None, None

artifact, model_path = load_trained_pipeline()

# Extract pipeline and metadata
if artifact is not None:
    if isinstance(artifact, dict) and "pipeline" in artifact:
        pipeline = artifact["pipeline"]
        feature_names = artifact.get("feature_names", [
            'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
            'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
        ])
        class_labels = artifact.get("class_labels", {0: "Non-Potable", 1: "Potable"})
        training_acc = artifact.get("training_cv_accuracy", 0.6599)
        test_acc = artifact.get("test_accuracy", 0.6402)
    else:
        pipeline = artifact
        feature_names = [
            'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
            'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
        ]
        class_labels = {0: "Non-Potable", 1: "Potable"}
        training_acc = 0.6599
        test_acc = 0.6402
else:
    pipeline = None
    feature_names = []
    class_labels = {}
    training_acc = 0.0
    test_acc = 0.0

# ---------------------------------------------------------
# Sidebar: Project Info & Mode Selector
# ---------------------------------------------------------
st.sidebar.title("💧 Project ML_17")
st.sidebar.markdown("**IoT Water Quality Safety Classifier**")
st.sidebar.markdown("*Department of Computer Science & Engineering*")
st.sidebar.markdown("---")

if model_path:
    st.sidebar.success(f"✓ Model Loaded:\n`{model_path.name}`")
    st.sidebar.caption(f"**Algorithm:** Tuned k-NN (k=25, distance, euclidean)")
    st.sidebar.caption(f"**5-Fold CV Accuracy:** {training_acc:.2%}")
    st.sidebar.caption(f"**Held-Out Test Accuracy:** {test_acc:.2%}")
else:
    st.sidebar.error("❌ Model artifact not found. Please train via notebook first.")

st.sidebar.markdown("---")
app_mode = st.sidebar.radio(
    "Select Operating Mode:",
    ["🔬 Interactive Predictor", "📡 IoT Sensor Guide & Architecture", "📊 Model Card & Lab Specs"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Academic Demonstration Note:**\n"
    "This app demonstrates ML model serving with preprocessed pipelines. "
    "Predictions are for educational study only."
)

# ---------------------------------------------------------
# Main Page: Mode 1 - Interactive Predictor
# ---------------------------------------------------------
if app_mode == "🔬 Interactive Predictor":
    st.title("💧 IoT Water Quality Safety Classifier")
    st.markdown(
        "Evaluate water sample potability using physical, chemical, and simulated IoT sensor inputs. "
        "The model preprocesses inputs with median imputation and standard scaling, followed by a tuned k-NN classifier."
    )

    if pipeline is None:
        st.error(
            "⚠️ Pre-trained model pipeline not found! Please run the Jupyter notebook "
            "`notebooks/water_quality_project.ipynb` to train and save the pipeline."
        )
        st.stop()

    st.markdown("### Choose Input Source")
    input_source = st.radio(
        "Input Method:",
        ["✍️ Manual Entry (Laboratory / Custom)", "⚡ Preset Simulated IoT Water Bodies"],
        horizontal=True
    )

    # Defaults dictionary (based on dataset medians)
    default_vals = {
        'ph': 7.04,
        'Hardness': 196.97,
        'Solids': 20927.83,
        'Chloramines': 7.13,
        'Sulfate': 333.07,
        'Conductivity': 421.88,
        'Organic_carbon': 14.22,
        'Trihalomethanes': 66.62,
        'Turbidity': 3.96
    }

    if input_source == "⚡ Preset Simulated IoT Water Bodies":
        st.markdown(
            "> [!NOTE]\n"
            "> **Notice on Simulated IoT Readings:** These presets simulate readings collected from field telemetry. "
            "They are synthetic educational values and do not represent live physical hardware streams."
        )
        preset = st.selectbox(
            "Select a Simulated Water Body Profile:",
            [
                "Profile A: Clean Municipal Treated Water (Expected Potable)",
                "Profile B: Industrial / Agricultural Runoff (Expected Non-Potable)",
                "Profile C: Neutral Groundwater Source (Borderline Quality)",
                "Profile D: Random Field Sensor Telemetry"
            ]
        )

        if "Profile A" in preset:
            default_vals = {
                'ph': 7.30, 'Hardness': 175.0, 'Solids': 16500.0, 'Chloramines': 6.80,
                'Sulfate': 305.0, 'Conductivity': 390.0, 'Organic_carbon': 11.5,
                'Trihalomethanes': 58.0, 'Turbidity': 2.80
            }
        elif "Profile B" in preset:
            default_vals = {
                'ph': 5.20, 'Hardness': 260.0, 'Solids': 38000.0, 'Chloramines': 10.50,
                'Sulfate': 440.0, 'Conductivity': 620.0, 'Organic_carbon': 22.0,
                'Trihalomethanes': 105.0, 'Turbidity': 5.90
            }
        elif "Profile C" in preset:
            default_vals = {
                'ph': 6.85, 'Hardness': 210.0, 'Solids': 23000.0, 'Chloramines': 7.40,
                'Sulfate': 340.0, 'Conductivity': 440.0, 'Organic_carbon': 14.8,
                'Trihalomethanes': 72.0, 'Turbidity': 4.10
            }
        else: # Random Field Sensor Telemetry
            default_vals = {
                'ph': round(np.random.uniform(5.5, 9.0), 2),
                'Hardness': round(np.random.uniform(120.0, 280.0), 2),
                'Solids': round(np.random.uniform(10000.0, 35000.0), 1),
                'Chloramines': round(np.random.uniform(4.0, 10.0), 2),
                'Sulfate': round(np.random.uniform(250.0, 420.0), 1),
                'Conductivity': round(np.random.uniform(250.0, 600.0), 1),
                'Organic_carbon': round(np.random.uniform(8.0, 20.0), 2),
                'Trihalomethanes': round(np.random.uniform(30.0, 95.0), 1),
                'Turbidity': round(np.random.uniform(2.0, 5.8), 2)
            }

    st.markdown("#### Water Quality Measurement Inputs")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Physical & Electro-chemical")
        in_ph = st.number_input(
            "1. pH Value (0 - 14)",
            min_value=0.0, max_value=14.0, value=float(default_vals['ph']), step=0.05,
            help="WHO standard: 6.5 to 8.5. Measurable via analog glass-electrode pH sensor."
        )
        in_turb = st.number_input(
            "2. Turbidity (NTU)",
            min_value=0.0, max_value=10.0, value=float(default_vals['Turbidity']), step=0.05,
            help="Optical clarity of water. Measurable via optical light-scattering turbidity sensor."
        )
        in_cond = st.number_input(
            "3. Conductivity (μS/cm)",
            min_value=50.0, max_value=1000.0, value=float(default_vals['Conductivity']), step=1.0,
            help="Electrical conductivity influenced by dissolved ions. Measurable via EC/TDS sensor probe."
        )

    with col2:
        st.subheader("Dissolved Solids & Hardness")
        in_solids = st.number_input(
            "4. Total Dissolved Solids / TDS (ppm)",
            min_value=100.0, max_value=70000.0, value=float(default_vals['Solids']), step=100.0,
            help="Total mineral and salt content. Estimated via TDS sensor."
        )
        in_hard = st.number_input(
            "5. Hardness (mg/L CaCO₃)",
            min_value=10.0, max_value=400.0, value=float(default_vals['Hardness']), step=1.0,
            help="Soap precipitation capacity. Requires chemical titration in laboratory."
        )
        in_sulfate = st.number_input(
            "6. Sulfate (mg/L)",
            min_value=50.0, max_value=600.0, value=float(default_vals['Sulfate']), step=1.0,
            help="Dissolved sulfate salts. Requires spectrophotometric lab assay."
        )

    with col3:
        st.subheader("Disinfectants & Organic Chemistry")
        in_chlor = st.number_input(
            "7. Chloramines (ppm)",
            min_value=0.0, max_value=20.0, value=float(default_vals['Chloramines']), step=0.05,
            help="Chlorine + ammonia municipal disinfectant. Requires laboratory colorimetric test."
        )
        in_org = st.number_input(
            "8. Organic Carbon / TOC (ppm)",
            min_value=0.0, max_value=40.0, value=float(default_vals['Organic_carbon']), step=0.1,
            help="Total organic carbon. Requires high-temperature catalytic combustion analyzer."
        )
        in_thm = st.number_input(
            "9. Trihalomethanes / THM (μg/L)",
            min_value=0.0, max_value=150.0, value=float(default_vals['Trihalomethanes']), step=0.5,
            help="Disinfection byproduct. Requires laboratory Gas Chromatography-Mass Spectrometry (GC-MS)."
        )

    # Build input DataFrame
    input_data = pd.DataFrame([{
        'ph': in_ph,
        'Hardness': in_hard,
        'Solids': in_solids,
        'Chloramines': in_chlor,
        'Sulfate': in_sulfate,
        'Conductivity': in_cond,
        'Organic_carbon': in_org,
        'Trihalomethanes': in_thm,
        'Turbidity': in_turb
    }])[feature_names]

    st.markdown("---")
    predict_clicked = st.button("🔍 Classify Water Quality", type="primary", use_container_width=True)

    if predict_clicked:
        with st.spinner("Executing pipeline (Imputation → Standardization → k-NN Distance Voting)..."):
            prediction = pipeline.predict(input_data)[0]
            probabilities = pipeline.predict_proba(input_data)[0]

        st.subheader("Classification Outcome")
        res_col1, res_col2 = st.columns([1.2, 1])

        with res_col1:
            if prediction == 1:
                st.success("### ✅ Predicted Potable (Safe to Drink)")
                st.markdown(
                    "The model's nearest neighbors classify this sample as **Potable** "
                    "(meeting the historical baseline patterns for safe drinking water in the dataset)."
                )
            else:
                st.error("### 🚫 Predicted Non-Potable (Unsafe to Drink)")
                st.markdown(
                    "The model's nearest neighbors classify this sample as **Non-Potable** "
                    "(exhibiting characteristics associated with contaminated or non-potable water bodies)."
                )

            # High-visibility Academic Safety Disclaimer
            st.warning(
                "⚠️ **MANDATORY SAFETY DISCLAIMER:**\n\n"
                "This prediction is generated by an educational machine learning model (Project ML_17). "
                "**It CANNOT certify real-world water safety.** Genuine potable water certification "
                "strictly requires comprehensive biological testing (e.g. testing for *E. coli*, coliform bacteria, "
                "viruses, and toxic heavy metals) performed by certified environmental laboratories."
            )

        with res_col2:
            st.markdown("#### Model Scoring Breakdown")
            prob_non_potable = probabilities[0]
            prob_potable = probabilities[1]

            st.write(f"**Non-Potable Score:** `{prob_non_potable * 100:.1f}%`")
            st.progress(float(prob_non_potable))

            st.write(f"**Potable Score:** `{prob_potable * 100:.1f}%`")
            st.progress(float(prob_potable))

            st.caption(
                "ℹ️ **Note on Scoring:** These values reflect distance-weighted neighbor proportions "
                "from the k-NN algorithm (`predict_proba`). They represent mathematical distance in feature space, "
                "**not a guaranteed probability or physical certainty of safety.**"
            )

        with st.expander("🔎 View Normalized Input Vector Sent to Pipeline"):
            st.dataframe(input_data, use_container_width=True)

# ---------------------------------------------------------
# Main Page: Mode 2 - IoT Sensor Guide & Architecture
# ---------------------------------------------------------
elif app_mode == "📡 IoT Sensor Guide & Architecture":
    st.title("📡 IoT Water Quality System Architecture")
    st.markdown(
        "A realistic engineering analysis of how IoT edge devices connect to machine learning pipelines, "
        "and which features can genuinely be monitored via field sensors versus laboratory instruments."
    )

    st.markdown("### 1. Feature Feasibility Matrix: Real Sensors vs Lab Instruments")
    st.markdown(
        "A common misconception in student projects is assuming a microcontroller like Arduino or ESP32 "
        "can measure complex organic chemicals or disinfection byproducts. The table below outlines the real-world boundary:"
    )

    feasibility_data = [
        {
            "Feature": "pH",
            "Measurement Type": "IoT Field Sensor",
            "Hardware Sensor": "Analog pH Electrode (e.g., pH-4502C Sensor Module)",
            "Deployment Feasibility": "High (Readily available, $15-$30)"
        },
        {
            "Feature": "Turbidity",
            "Measurement Type": "IoT Field Sensor",
            "Hardware Sensor": "Optical Light Scattering Module (e.g., DFRobot Turbidity Sensor)",
            "Deployment Feasibility": "High (Readily available, $10-$25)"
        },
        {
            "Feature": "Conductivity",
            "Measurement Type": "IoT Field Sensor",
            "Hardware Sensor": "Analog Electrical Conductivity (EC) Probe",
            "Deployment Feasibility": "High (Readily available, $15-$30)"
        },
        {
            "Feature": "Solids (TDS)",
            "Measurement Type": "IoT Field Sensor",
            "Hardware Sensor": "Analog TDS Meter (Calculated from EC)",
            "Deployment Feasibility": "High (Readily available, $10-$20)"
        },
        {
            "Feature": "Chloramines",
            "Measurement Type": "Laboratory Analysis",
            "Hardware Sensor": "DPD Colorimetric Reagents / Spectrophotometer",
            "Deployment Feasibility": "Not feasible for low-cost microcontroller IoT"
        },
        {
            "Feature": "Sulfate",
            "Measurement Type": "Laboratory Analysis",
            "Hardware Sensor": "Barium Chloride Turbidimetry / Ion Chromatography",
            "Deployment Feasibility": "Requires chemical wet lab reagents"
        },
        {
            "Feature": "Organic Carbon (TOC)",
            "Measurement Type": "Laboratory Analysis",
            "Hardware Sensor": "High-Temperature Catalytic Combustion / NDIR Detector",
            "Deployment Feasibility": "Requires specialized $10,000+ benchtop analyzer"
        },
        {
            "Feature": "Trihalomethanes (THM)",
            "Measurement Type": "Laboratory Analysis",
            "Hardware Sensor": "Gas Chromatography-Mass Spectrometry (GC-MS)",
            "Deployment Feasibility": "Requires accredited environmental laboratory"
        },
        {
            "Feature": "Hardness",
            "Measurement Type": "Laboratory Analysis",
            "Hardware Sensor": "EDTA Complexometric Titration",
            "Deployment Feasibility": "Requires manual or automated burette titration"
        }
    ]
    st.dataframe(pd.DataFrame(feasibility_data), use_container_width=True)

    st.markdown("---")
    st.markdown("### 2. Physical IoT Hardware Integration Blueprint (Optional Extension)")
    st.markdown(
        """
```
+-----------------------------------------------------------------------------------+
|                            PHYSICAL IOT EDGE NODE                                 |
|                                                                                   |
|  +--------------------+   Analog (A0)                                             |
|  | Analog pH Probe    | -----------------+                                        |
|  +--------------------+                  |                                        |
|                                          v                                        |
|  +--------------------+   Analog (A1)  +----------------------+                   |
|  | Turbidity Sensor   | -------------> | ESP32 Microcontroller|                   |
|  +--------------------+                | (12-bit ADC, WiFi)   |                   |
|                                        +----------------------+                   |
|  +--------------------+   Analog (A2)            |                                |
|  | Analog TDS Probe   | -------------------------+                                |
|  +--------------------+                          | MQTT / HTTP REST POST          |
+--------------------------------------------------|--------------------------------+
                                                   v
                                 +-----------------------------------+
                                 | Cloud / Edge Server (Streamlit)   |
                                 | - Receives live pH, Turbidity, TDS|
                                 | - Merges lab parameters/imputations|
                                 | - Runs scikit-learn k-NN pipeline |
                                 | - Publishes Safety Status Alert   |
                                 +-----------------------------------+
```
        """
    )

    st.markdown("### 3. Sensor Calibration Requirements for Viva & Practical Defense")
    st.markdown(
        """
- **pH Sensor Calibration:** Must be calibrated using a standard two-point buffer solution (pH 4.01 acidic buffer and pH 6.86 / 7.00 neutral buffer) before taking measurements. Temperature compensation is required because pH voltage varies with temperature.
- **Turbidity Sensor Calibration:** Must be zero-calibrated in distilled (deionized) water at 0.0 NTU, and checked against a Formazin turbidity standard solution.
- **TDS / Conductivity Calibration:** Must be calibrated against a known electrical conductivity standard solution (typically 1413 μS/cm KCl solution).
        """
    )

# ---------------------------------------------------------
# Main Page: Mode 3 - Model Card & Lab Specs
# ---------------------------------------------------------
elif app_mode == "📊 Model Card & Lab Specs":
    st.title("📊 Model Card & Experimental Specifications")
    st.markdown("Comprehensive documentation of the machine learning experiments conducted for Project ID: ML_17.")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Model Configuration")
        st.markdown(
            """
- **Primary Algorithm:** k-Nearest Neighbors (k-NN)
- **Comparison Algorithm:** Gaussian Naive Bayes (GaussianNB)
- **Baseline:** DummyClassifier (Most Frequent strategy)
- **Pipeline Architecture:**
  1. `SimpleImputer(strategy='median')`
  2. `StandardScaler()`
  3. `KNeighborsClassifier(n_neighbors=25, weights='distance', metric='euclidean')`
- **Validation Strategy:** Stratified 5-Fold Cross-Validation on 80% training data (2,620 samples).
- **Held-Out Test Set:** 20% preserved untouched until final reporting (656 samples).
            """
        )

    with col_b:
        st.subheader("Experimental Results (Verified)")
        res_table = pd.DataFrame({
            "Experiment": [
                "Dummy Baseline (CV)",
                "Gaussian Naive Bayes (CV)",
                "Tuned k-NN (5-Fold CV)",
                "Tuned k-NN (Held-Out Test Set)"
            ],
            "Accuracy": ["60.99%", "62.75%", "65.99%", "64.02%"],
            "Role": ["Baseline Benchmark", "Analytical Comparison", "Model Selection Metric", "Final Generalization Metric"]
        })
        st.table(res_table)

    st.markdown("### Error Distribution on Held-Out Test Set (656 samples)")
    err_col1, err_col2, err_col3, err_col4 = st.columns(4)
    err_col1.metric("True Negatives (TN)", "368", "Correct Unsafe")
    err_col2.metric("False Positives (FP)", "32", "Dangerous Error")
    err_col3.metric("False Negatives (FN)", "204", "Cautious Error")
    err_col4.metric("True Positives (TP)", "52", "Correct Safe")

    st.markdown(
        """
> [!IMPORTANT]
> **Key Academic Takeaway:**
> The model achieves **64.02% test accuracy**, significantly outperforming the naive baseline in discriminating between safe and unsafe bodies. The False Positive rate is kept low (32 samples, ~4.8% of total test set), which is crucial for public health risk mitigation.
        """
    )

# Footer
st.markdown("---")
st.caption(
    "IoT Water Quality Safety Classifier | Project ID: ML_17 | "
    "Built with Python, scikit-learn, and Streamlit."
)

