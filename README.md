# IoT Water Quality Safety Classifier (Project ML_17)

An end-to-end machine learning and IoT demonstration project that classifies water potability (safe vs. unsafe drinking water) from physical, chemical, and simulated sensor parameters using a leak-free **k-Nearest Neighbors (k-NN)** pipeline and **Gaussian Naive Bayes (GaussianNB)** comparison model.

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Details & License](#dataset-details--license)
3. [Repository Structure](#repository-structure)
4. [Step-by-Step Setup Guide (Windows)](#step-by-step-setup-guide-windows)
5. [Running the Jupyter Notebook](#running-the-jupyter-notebook)
6. [Running the Streamlit Web App](#running-the-streamlit-web-app)
7. [Deploying to Streamlit Community Cloud](#deploying-to-streamlit-community-cloud)
8. [Experimental Results & Verification](#experimental-results--verification)
9. [IoT Sensor Feasibility & Calibration Guide](#iot-sensor-feasibility--calibration-guide)
10. [Project Report Outline](#project-report-outline)
11. [10 Viva Voce Questions & Simple Answers](#10-viva-voce-questions--simple-answers)

---

## 1. Project Overview

- **Title:** IoT Water Quality Safety Classifier
- **Project ID:** ML_17
- **Primary Model:** k-Nearest Neighbors (`KNeighborsClassifier`) with hyperparameter tuning via Stratified 5-Fold Cross-Validation (`GridSearchCV`).
- **Benchmark Models:** `DummyClassifier` (majority-class baseline) and `GaussianNB` (Gaussian Naive Bayes).
- **Core Engineering:** End-to-end leak-free `scikit-learn` Pipeline combining `SimpleImputer(strategy='median')`, `StandardScaler()`, and the classifier.
- **Serving:** Interactive `Streamlit` dashboard supporting manual input, simulated IoT sensor telemetry presets, hardware connection blueprints, and educational safety disclaimers.

---

## 2. Dataset Details & License

- **Source:** Kaggle Water Quality / Water Potability Dataset
- **Author:** Aditya Kadiwal
- **License:** **CC0: Public Domain** (Open access, suitable for educational research and reproduction)
- **Target Variable:** `Potability` (`1` = Potable / Safe to Drink, `0` = Non-Potable / Unsafe to Drink)
- **Dataset Size:** 3,276 samples, 10 columns (9 input features + 1 target)
- **Class Distribution:** Non-potable: 1,998 (~60.99%), Potable: 1,278 (~39.01%)

### Feature Description & Scientific Units:
| Feature | Scientific Unit | Typical Clean Range | Measurement Source |
| :--- | :--- | :--- | :--- |
| `ph` | Dimensionless (0–14) | 6.5 – 8.5 (WHO) | Analog pH Electrode Sensor |
| `Hardness` | mg/L as $\text{CaCO}_3$ | < 200 mg/L | Laboratory EDTA Titration |
| `Solids` | ppm (Total Dissolved Solids) | 500 – 1000 ppm | Analog TDS / EC Sensor |
| `Chloramines` | ppm | < 4 ppm (EPA) | Colorimetric Lab Assay |
| `Sulfate` | mg/L | < 250 mg/L | Spectrophotometry Lab Test |
| `Conductivity` | $\mu\text{S/cm}$ | < 400 $\mu\text{S/cm}$ | Electrical Conductivity Probe |
| `Organic_carbon` | ppm (Total Organic Carbon) | < 2 ppm | High-Temp Catalytic Combustion |
| `Trihalomethanes` | $\mu\text{g/L}$ (ppb) | < 80 $\mu\text{g/L}$ | Gas Chromatography-Mass Spec |
| `Turbidity` | NTU (Cloudiness) | < 5 NTU | Optical Turbidity Sensor |

---

## 3. Repository Structure

```
water-quality-classifier/
│
├── data/
│   └── water_potability.csv          # Kaggle dataset CSV
│
├── notebooks/
│   └── water_quality_project.ipynb   # Complete step-by-step ML pipeline notebook
│
├── models/
│   └── water_quality_pipeline.joblib # Serialized fitted pipeline + metadata
│
├── app.py                            # Streamlit web application & IoT simulator
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation, report outline & viva
└── .gitignore                        # Git exclusion rules (secrets, venv, cache)
```

---

## 4. Step-by-Step Setup Guide (Windows)

Open **PowerShell** or **Command Prompt** and follow these commands:

### Step 4.1: Navigate to your Project Directory
```powershell
cd "c:\Users\PC\Downloads\water quality ml\water-quality-classifier"
```

### Step 4.2: Create a Virtual Environment
```powershell
python -m venv .venv
```

### Step 4.3: Activate the Virtual Environment
- **In PowerShell:**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
  *(If execution policy restricts scripts, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*
- **In Command Prompt (cmd):**
  ```cmd
  .venv\Scripts\activate.bat
  ```

### Step 4.4: Install Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. Running the Jupyter Notebook

To launch the notebook and re-train or inspect the pipeline:

```powershell
pip install jupyter
jupyter notebook notebooks/water_quality_project.ipynb
```

In Jupyter Notebook:
1. Click **Cell** $\rightarrow$ **Run All** (or run each cell sequentially with `Shift + Enter`).
2. The notebook executes data inspection, exploratory data analysis plots, leak-free pipeline creation, 5-fold cross-validation, test evaluation, and saves the pipeline to `models/water_quality_pipeline.joblib`.

---

## 6. Running the Streamlit Web App

To start the interactive web application locally:

```powershell
streamlit run app.py
```

The application will launch in your default web browser at `http://localhost:8501`.

---

## 7. Deploying to Streamlit Community Cloud

Follow these verified steps to deploy your live project online for free:

1. **Initialize a Git repository:**
   ```powershell
   git init
   git add .
   git commit -m "Initial commit for IoT Water Quality Classifier (ML_17)"
   ```
2. **Push to GitHub:**
   - Create a new public repository on [GitHub](https://github.com).
   - Link and push your local branch:
     ```powershell
     git remote add origin https://github.com/<your-username>/water-quality-classifier.git
     git branch -M main
     git push -u origin main
     ```
3. **Deploy on Streamlit Community Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
   - Click **"New app"**.
   - Select your repository (`water-quality-classifier`), branch (`main`), and set **Main file path** to `app.py`.
   - Click **"Deploy!"**.
4. **Security & Licensing Best Practices:**
   - Ensure `.gitignore` ignores `kaggle.json` or private tokens.
   - The dataset has a **CC0: Public Domain** license, permitting public hosting.

---

## 8. Experimental Results & Verification

All numbers below reflect actual model execution on the verified dataset without fabrication:

| Stage | Model / Strategy | Accuracy | F1-Score (Class 1) | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | DummyClassifier (Most Frequent) | 60.99% | 0.0000 | Always predicts majority class (0) |
| **Comparison** | Gaussian Naive Bayes (5-Fold CV) | 62.75% | 0.3812 | Analytical Gaussian probabilistic model |
| **Main Model** | Tuned k-NN (5-Fold CV) | **65.99%** | **0.4280** | Best params: $k=25$, distance, euclidean |
| **Held-Out Test** | Tuned k-NN Pipeline (Final Test) | **64.02%** | **0.3059** | Evaluated on 656 unseen test samples |

### Test Set Confusion Matrix (656 samples):
- **True Negatives (TN):** `368` (Correctly predicted Non-Potable)
- **False Positives (FP):** `32` (Contaminated water predicted Potable — **Safety Hazard**)
- **False Negatives (FN):** `204` (Potable water predicted Non-Potable — **Cautious Error**)
- **True Positives (TP):** `52` (Correctly predicted Potable)

---

## 9. IoT Sensor Feasibility & Calibration Guide

### Practical IoT vs. Chemistry Laboratory Reality
- **IoT Deployable Sensors:**
  - `pH`: Analog glass-electrode sensor probe (e.g. pH-4502C).
  - `Turbidity`: Optical photodiode scattering sensor (e.g. DFRobot Analog Turbidity Sensor).
  - `Conductivity` & `Solids`: Dual-needle electrical conductivity probe.
- **Laboratory-Only Parameters:**
  - `Chloramines`: Requires colorimetric reagent photometer.
  - `Sulfate`: Requires barium chloride precipitation or ion chromatography.
  - `Organic_carbon`: Requires high-temperature catalytic combustion ($>680^\circ\text{C}$).
  - `Trihalomethanes`: Requires Gas Chromatography-Mass Spectrometry (GC-MS).
  - `Hardness`: Requires EDTA chemical titration.

### Sensor Calibration Steps:
1. **pH Sensor:** Perform 2-point calibration in standard buffer solutions (pH 4.01 and pH 6.86/7.00). Use temperature compensation formula $V_{\text{actual}} = V_0 + k(T - 25)$.
2. **Turbidity Sensor:** Calibrate 0 NTU baseline in distilled water. Calibrate high span against Formazin turbidity standard.
3. **EC/TDS Sensor:** Calibrate in 1413 $\mu\text{S/cm}$ standard KCl solution.

---

## 10. Project Report Outline

Use this outline for your college project documentation and report submission:

1. **Title Page & Certificates** (Project ID: ML_17, Student Names, Faculty Guide).
2. **Abstract** (Problem statement, potable water importance, k-NN & Naive Bayes methodology, Streamlit interface).
3. **Introduction & Literature Review** (Traditional water testing vs IoT ML classification).
4. **Dataset Analysis & Preprocessing** (Kaggle CC0 dataset, 9 features, median imputation, standardization, leakage prevention).
5. **System Architecture & Methodology** (Stratified splitting, scikit-learn Pipeline, k-NN distance weighting, GaussianNB comparison).
6. **Experimental Results & Discussion** (5-Fold CV tuning, confusion matrix breakdown, Type I vs Type II errors, real-world modest accuracy context).
7. **IoT Hardware Feasibility & Sensor Integration** (ESP32 telemetry, analog sensor interfaces, field vs lab distinction).
8. **Web Application & Deployment** (Streamlit architecture, model serialization with joblib, UI design).
9. **Conclusion & Future Scope** (Integration of microbiological sensors, edge ML deployment on microcontroller).
10. **References & Appendices**.

---

## 11. 10 Viva Voce Questions & Simple Answers

### Q1: Why do we use k-Nearest Neighbors (k-NN) for water potability?
**Answer:** k-NN is an intuitive, non-parametric algorithm that classifies a water sample by looking at the known labels of the $k$ most geometrically similar water samples in multi-dimensional feature space. It naturally captures non-linear relationships without making strong assumptions about data distribution.

### Q2: Why is Feature Scaling (`StandardScaler`) mandatory before running k-NN?
**Answer:** k-NN computes Euclidean distance ($\sqrt{\sum (x_i - y_i)^2}$). In our dataset, `Solids` reaches up to 60,000 while `ph` is between 0 and 14. Without scaling, `Solids` would mathematically dominate 99.9% of the distance calculation, making features like pH and turbidity completely irrelevant. `StandardScaler` places all features on equal footing with mean 0 and variance 1.

### Q3: What is "Data Leakage" and how did you prevent it?
**Answer:** Data leakage happens when information from the test dataset influences the training process (e.g. computing the mean or median of the whole dataset before splitting). We prevented leakage by performing a Stratified train/test split first and encapsulating imputation and scaling inside a `scikit-learn Pipeline`, so preprocessing parameters are learned strictly from training folds.

### Q4: Why did you tune $k$ using Stratified 5-Fold Cross-Validation rather than the test set?
**Answer:** If we tune hyperparameters on the test set, the model overfits to the test set, creating an overly optimistic evaluation. Stratified 5-Fold Cross-Validation evaluates candidate parameters across internal training folds while preserving class balance, leaving the test set completely pristine for an unbiased final evaluation.

### Q5: How did Gaussian Naive Bayes compare with k-NN?
**Answer:** Gaussian Naive Bayes achieved ~62.75% 5-fold CV accuracy, whereas tuned k-NN achieved ~65.99%. GaussianNB assumes all features are conditionally independent given the class, which is partially violated because physical and chemical water properties (like conductivity and dissolved solids) are inter-related. k-NN performed better by considering multi-dimensional proximity.

### Q6: What is a False Positive in this project, and why is it dangerous?
**Answer:** A False Positive occurs when the model predicts non-potable water is "Potable" (Safe). This is dangerous because human beings might consume contaminated water based on the prediction, leading to waterborne diseases or poisoning. In public health ML, minimizing False Positives is critical.

### Q7: What is a False Negative, and what is its consequence?
**Answer:** A False Negative occurs when clean, potable water is predicted to be "Non-Potable" (Unsafe). Its consequence is economic inconvenience—water is discarded or needlessly re-boiled or re-filtered, but it poses no danger to human life.

### Q8: Can an ESP32 or Arduino read all 9 features from the dataset using physical sensors?
**Answer:** No. An ESP32 can readily read physical and electrical properties (`pH`, `Turbidity`, `Conductivity`, and estimated `Solids`) using analog sensor probes. However, features like `Trihalomethanes`, `Organic Carbon`, `Chloramines`, `Sulfate`, and `Hardness` require chemical titration, spectrophotometry, or gas chromatography in a specialized analytical laboratory.

### Q9: Why is the model accuracy around ~64% rather than 95%+?
**Answer:** Real-world tabular water quality data has high overlap between potable and non-potable samples across physical-chemical measurements alone. Safe drinking water standards also depend heavily on microbiological pathogens (such as bacteria and viruses) that are not present in this physical-chemical tabular dataset. Reporting actual ~64% accuracy reflects genuine scientific integrity rather than synthetic overfitted results.

### Q10: Why did you save the model using a `Pipeline` inside `joblib`?
**Answer:** Saving the entire `Pipeline` ensures that when new raw measurements enter the Streamlit app, they automatically pass through the exact same median imputation and standard scaling parameters fitted during training, eliminating manual preprocessing code and preventing deployment mismatch bugs.

