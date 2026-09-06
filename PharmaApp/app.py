from __future__ import annotations

import io
import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from scipy.optimize import differential_evolution

from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import (
    train_test_split,
    RepeatedKFold,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Virtual Formulation Lab | Final Project",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed",
)

RANDOM_SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
CV_REPEATS = 2
MAX_OPT_VARS = 12

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "saved_cqa_models"
MODEL_DIR.mkdir(exist_ok=True)

DATASET_CANDIDATES = [
    BASE_DIR / "final Data All Exipients.csv",
    BASE_DIR / "final Data All Exipients.xlsx",
    BASE_DIR / "final Data All Exipients.xls",
]

TARGETS = [
    "HARDNESS",
    "FRIABILITY",
    "Drug content",
    "Water absorption ratio",
    "DISINTEGRATION_TIME",
]


# ============================================================
# PROFESSIONAL SCIENTIFIC UI STYLING
# ============================================================

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.1rem;
        letter-spacing: -0.5px;
    }

    .subtitle {
        color: #475569;
        font-size: 1rem;
        margin-bottom: 1.5rem;
        font-weight: 500;
    }

    .section-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-top: 1.2rem;
        margin-bottom: 0.6rem;
        border-bottom: 2px solid #cbd5e1;
        padding-bottom: 0.3rem;
    }

    .result-card {
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        background: #ffffff;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        text-align: center;
        transition: transform 0.2s;
    }
    
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px -1px rgba(0, 0, 0, 0.08);
    }

    .result-label {
        color: #64748b;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }

    .result-value {
        font-size: 1.4rem;
        font-weight: 800;
        color: #0284c7;
    }

    .info-box {
        padding: 12px 16px;
        border-radius: 8px;
        background: #f0f9ff;
        border-left: 4px solid #0284c7;
        color: #0369a1;
        font-size: 0.92rem;
        margin: 10px 0;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA HANDLING & MODEL FUNCTIONS
# ============================================================

@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    for path in DATASET_CANDIDATES:
        if path.exists():
            if path.suffix.lower() == ".csv":
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path)
            return clean_dataset(df)
    raise FileNotFoundError("Required formulation dataset was not found.")

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            converted = pd.to_numeric(
                df[col].astype(str).str.replace(",", "", regex=False),
                errors="coerce",
            )
            if converted.notna().mean() >= 0.80:
                df[col] = converted

    missing_targets = [c for c in TARGETS if c not in df.columns]
    if missing_targets:
        raise ValueError(f"Required CQA columns are missing: {missing_targets}")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    predictors = [c for c in numeric_cols if c not in TARGETS]
    
    df = df[predictors + TARGETS].replace([np.inf, -np.inf], np.nan).drop_duplicates()
    df = df.dropna(subset=TARGETS, how="all")
    return df

def calculate_metrics(y_true, y_pred) -> Dict[str, float]:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(rmse),
    }

def get_model_candidates():
    return {
        "Extra Trees": (ExtraTreesRegressor(random_state=RANDOM_SEED, n_jobs=-1), {
            "model__n_estimators": [300, 500, 700],
            "model__max_depth": [None, 8, 16],
            "model__min_samples_split": [2, 4],
        }),
        "Random Forest": (RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1), {
            "model__n_estimators": [300, 500],
            "model__max_depth": [None, 8, 16],
        }),
        "Gradient Boosting": (GradientBoostingRegressor(random_state=RANDOM_SEED), {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
        }),
        "HistGradient Boosting": (HistGradientBoostingRegressor(random_state=RANDOM_SEED), {
            "model__max_iter": [100, 200],
        }),
        "SVR": (SVR(kernel="rbf"), {
            "model__C": [1, 10, 100],
            "model__gamma": ["scale", "auto"],
        }),
        "Ridge": (Ridge(), {"model__alpha": [0.1, 1, 10]}),
    }

def make_pipeline(model):
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("feature_selection", SelectKBest(score_func=mutual_info_regression, k="all")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])

def train_one_cqa(df: pd.DataFrame, target: str):
    X = df.drop(columns=TARGETS)
    y = df[target]
    valid = y.notna()
    X, y = X.loc[valid].copy(), y.loc[valid].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    cv = RepeatedKFold(n_splits=CV_FOLDS, n_repeats=CV_REPEATS, random_state=RANDOM_SEED)
    candidates = get_model_candidates()

    best_pipeline, best_cv_score = None, -np.inf

    for model_name, (model, params) in candidates.items():
        pipeline = make_pipeline(model)
        search = RandomizedSearchCV(pipeline, params, n_iter=5, scoring="r2", cv=cv, random_state=RANDOM_SEED, n_jobs=-1, error_score=np.nan)
        try:
            search.fit(X_train, y_train)
            score = search.best_score_
            if np.isfinite(score) and score > best_cv_score:
                best_cv_score = score
                best_pipeline = search.best_estimator_
        except Exception:
            continue

    y_pred = best_pipeline.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)
    metrics["CV R2"] = float(best_cv_score)

    try:
        perm = permutation_importance(best_pipeline, X_train, y_train, scoring="r2", n_repeats=5, random_state=RANDOM_SEED, n_jobs=-1)
        importance = pd.DataFrame({"Feature": X_train.columns, "Importance": np.maximum(perm.importances_mean, 0)})
        total = importance["Importance"].sum()
        importance["Importance_%"] = (importance["Importance"] / total * 100) if total > 0 else 0.0
        importance = importance.sort_values("Importance_%", ascending=False)
    except Exception:
        importance = pd.DataFrame(columns=["Feature", "Importance", "Importance_%"])

    return {
        "model": best_pipeline,
        "target": target,
        "features": X.columns.tolist(),
        "metrics": metrics,
        "importance": importance,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
    }

def model_path(target: str):
    return MODEL_DIR / f"{target.replace(' ', '_').replace('/', '_')}.joblib"

def save_artifact(target: str, artifact: Dict[str, Any]):
    joblib.dump(artifact, model_path(target))

def load_artifact(target: str):
    path = model_path(target)
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None

def all_models_available():
    return all(model_path(target).exists() for target in TARGETS)

def initialize_models():
    df = load_dataset()
    progress = st.progress(0)
    status = st.empty()
    for i, target in enumerate(TARGETS):
        status.info(f"Training machine learning model for CQA: {target}...")
        artifact = train_one_cqa(df, target)
        save_artifact(target, artifact)
        progress.progress((i + 1) / len(TARGETS))
    status.success("Models initialized successfully.")
    progress.empty()

@st.cache_resource(show_spinner=False)
def load_all_models():
    return {target: load_artifact(target) for target in TARGETS if model_path(target).exists()}

def ensure_models():
    if not all_models_available():
        with st.spinner("Initializing models for the first time..."):
            initialize_models()
        st.cache_resource.clear()
    return load_all_models()

def predict_cqas(models, inputs):
    return {target: float(models[target]["model"].predict(inputs)[0]) for target in TARGETS}

def get_input_ranges(df):
    ranges = {}
    for col in df.columns:
        if col in TARGETS:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) == 0:
            continue
        ranges[col] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "unique": series.nunique(),
            "values": sorted(series.unique().tolist())
        }
    return ranges

def classify_features(input_columns):
    # قائمة بأسماء الخصائص الفيزيائية والكيميائية الشائعة في الداتا
    physicochemical_keywords = [
        "Weight", "Density", "Index", "Ratio", "Repose", "Thickness", "Molecular", 
        "XLogP3", "Hydrogen", "Rotational", "Topological", "Heavy", "Complexity", "LogS", "DOSE", "Wetting"
    ]
    
    phys_cols = []
    excipient_cols = []
    
    for col in input_columns:
        if any(kw.lower() in col.lower() for kw in physicochemical_keywords):
            phys_cols.append(col)
        else:
            excipient_cols.append(col)
            
    return phys_cols, excipient_cols

def desirability(value, goal, low, high, target=None):
    if not np.isfinite(value) or high <= low:
        return 0.0
    if goal == "Minimize":
        return 1.0 if value <= low else (0.0 if value >= high else (high - value) / (high - low))
    if goal == "Maximize":
        return 1.0 if value >= high else (0.0 if value <= low else (value - low) / (high - low))
    target = float(np.clip(target if target is not None else (low + high) / 2, low, high))
    if value == target:
        return 1.0
    if value < target:
        return 1.0 if target == low else max(0.0, (value - low) / (target - low))
    return 1.0 if high == target else max(0.0, (high - value) / (high - target))

def overall_desirability(predictions, goals, df):
    values = [np.clip(desirability(predictions[t], goals[t]["goal"], float(df[t].min()), float(df[t].max()), goals[t].get("target")), 0, 1) for t in TARGETS]
    return float(np.prod(values) ** (1 / len(values))) if values else 0.0

def select_optimization_variables(models):
    score_map = {}
    for target in TARGETS:
        for _, row in models[target]["importance"].iterrows():
            score_map[row["Feature"]] = score_map.get(row["Feature"], 0) + float(row["Importance_%"])
    return [f for f, _ in sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:MAX_OPT_VARS]]

def optimize_formulation(df, models, baseline_inputs, goals):
    ranges = get_input_ranges(df)
    selected = [x for x in select_optimization_variables(models) if x in ranges]
    if not selected:
        return baseline_inputs.copy(), {}, 0.0
    base = baseline_inputs.copy()
    bounds = [(ranges[f]["min"], ranges[f]["max"]) for f in selected]
    disc = {f: ranges[f]["values"] for f in selected if ranges[f]["unique"] <= 10}

    def objective(vals):
        cand = base.copy()
        for f, v in zip(selected, vals):
            if f in disc:
                arr = np.array(disc[f])
                v = float(arr[np.argmin(np.abs(arr - v))])
            cand[f] = v
        return -overall_desirability(predict_cqas(models, cand), goals, df)

    res = differential_evolution(objective, bounds=bounds, seed=RANDOM_SEED, maxiter=50, popsize=8, polish=True, workers=1)
    
    opt_inputs = base.copy()
    for f, v in zip(selected, res.x):
        if f in disc:
            arr = np.array(disc[f])
            v = float(arr[np.argmin(np.abs(arr - v))])
        opt_inputs[f] = v
    opt_preds = predict_cqas(models, opt_inputs)
    return opt_inputs, opt_preds, overall_desirability(opt_preds, goals, df)


# ============================================================
# APP INITIALIZATION
# ============================================================

try:
    df = load_dataset()
    models = ensure_models()
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.stop()

input_columns = [c for c in df.columns if c not in TARGETS]
input_ranges = get_input_ranges(df)
phys_cols, excipient_cols = classify_features(input_columns)

# App Header
st.markdown('<div class="main-title">🧪 Virtual Formulation Lab</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Driven Pharmaceutical Formulation & Quality Prediction Platform (Final Project)</div>', unsafe_allow_html=True)

tab_lab, tab_optimizer, tab_insights = st.tabs([
    "🧪 Formulation Setup & Prediction",
    "🎯 AI Multi-Criteria Optimizer",
    "📊 Scientific Model Insights"
])


# ============================================================
# TAB 1 — FORMULATION LAB (Separated Excipients & Physico-Chemical)
# ============================================================

with tab_lab:
    st.markdown('<div class="section-title">Batch Configuration</div>', unsafe_allow_html=True)
    
    col_name, _ = st.columns([2, 1])
    with col_name:
        formulation_name = st.text_input("Formulation Batch Code", value="Batch_01")

    st.markdown('<div class="info-box">Configure the physical parameters and excipient quantities below. The AI model will evaluate the Critical Quality Attributes (CQAs) instantly.</div>', unsafe_allow_html=True)

    values = {}

    # قسم الخصائص الفيزيائية والكيميائية
    with st.expander("⚙️ Physicochemical & Powder Properties", expanded=True):
        if phys_cols:
            for start in range(0, len(phys_cols), 3):
                cols = st.columns(3)
                for col_idx, feature in enumerate(phys_cols[start:start + 3]):
                    info = input_ranges[feature]
                    min_v, max_v = info["min"], info["max"]
                    with cols[col_idx]:
                        if min_v == max_v:
                            values[feature] = min_v
                            st.number_input(f"{feature} (Fixed)", value=float(min_v), disabled=True, key=f"inp_{feature}")
                        elif info["unique"] <= 10:
                            values[feature] = st.selectbox(feature, options=info["values"], key=f"inp_{feature}")
                        else:
                            values[feature] = st.number_input(
                                feature,
                                min_value=float(min_v),
                                max_value=float(max_v),
                                value=float((min_v + max_v) / 2),
                                format="%.4f",
                                key=f"inp_{feature}"
                            )

    # قسم السواغات (Excipients) منفصل تماماً لتجنب الزحمة
    with st.expander("💊 Excipients Composition", expanded=False):
        if excipient_cols:
            for start in range(0, len(excipient_cols), 3):
                cols = st.columns(3)
                for col_idx, feature in enumerate(excipient_cols[start:start + 3]):
                    info = input_ranges[feature]
                    min_v, max_v = info["min"], info["max"]
                    with cols[col_idx]:
                        if min_v == max_v:
                            values[feature] = min_v
                            st.number_input(f"{feature} (Fixed)", value=float(min_v), disabled=True, key=f"exc_{feature}")
                        elif info["unique"] <= 10:
                            values[feature] = st.selectbox(feature, options=info["values"], key=f"exc_{feature}")
                        else:
                            values[feature] = st.number_input(
                                feature,
                                min_value=float(min_v),
                                max_value=float(max_v),
                                value=float(min_v),
                                format="%.4f",
                                key=f"exc_{feature}"
                            )

    if st.button("🔬 Predict Critical Quality Attributes (CQAs)", type="primary", use_container_width=True):
        st.session_state.baseline_inputs = pd.DataFrame([values], columns=input_columns)
        st.session_state.predictions = predict_cqas(models, st.session_state.baseline_inputs)

    if "predictions" in st.session_state and st.session_state.predictions:
        st.markdown('<div class="section-title">Predicted CQAs Output</div>', unsafe_allow_html=True)
        res_cols = st.columns(len(TARGETS))
        for i, target in enumerate(TARGETS):
            val = st.session_state.predictions[target]
            with res_cols[i]:
                st.markdown(f"""
                    <div class="result-card">
                        <div class="result-label">{target}</div>
                        <div class="result-value">{val:.3f}</div>
                    </div>
                """, unsafe_allow_html=True)


# ============================================================
# TAB 2 — AI OPTIMIZER
# ============================================================

with tab_optimizer:
    st.markdown('<div class="section-title">Optimization Target Settings</div>', unsafe_allow_html=True)
    goals = {}
    goal_cols = st.columns(2)
    
    for i, target in enumerate(TARGETS):
        with goal_cols[i % 2]:
            goal = st.selectbox(f"{target} Strategy", ["Target", "Minimize", "Maximize"], key=f"goal_{target}")
            target_val = None
            if goal == "Target":
                s = df[target].dropna()
                target_val = st.number_input(f"Target value for {target}", float(s.min()), float(s.max()), float(s.median()), key=f"tval_{target}")
            goals[target] = {"goal": goal, "target": target_val}

    if st.button("🚀 Run AI Formulation Optimizer", type="primary", use_container_width=True):
        if "baseline_inputs" not in st.session_state:
            st.warning("Please compute initial predictions first in the Formulation Lab tab.")
        else:
            with st.spinner("Running optimization algorithm..."):
                opt_in, opt_pred, des = optimize_formulation(df, models, st.session_state.baseline_inputs, goals)
                st.session_state.opt_inputs = opt_in
                st.session_state.opt_preds = opt_pred
                st.session_state.opt_des = des

    if "opt_preds" in st.session_state:
        st.markdown('<div class="section-title">Optimization Results</div>', unsafe_allow_html=True)
        st.metric("Overall Desirability Score", f"{st.session_state.opt_des:.3f}")
        
        comp_df = pd.DataFrame({
            "CQA": TARGETS,
            "Current Prediction": [st.session_state.predictions[t] for t in TARGETS],
            "Optimized Prediction": [st.session_state.opt_preds[t] for t in TARGETS]
        })
        st.dataframe(comp_df, use_container_width=True, hide_index=True)


# ============================================================
# TAB 3 — SCIENTIFIC INSIGHTS
# ============================================================

with tab_insights:
    st.markdown('<div class="section-title">Model Performance Metrics</div>', unsafe_allow_html=True)
    val_data = [{"CQA": t, "CV R²": models[t]["metrics"]["CV R2"], "Test R²": models[t]["metrics"]["R2"], "Test RMSE": models[t]["metrics"]["RMSE"]} for t in TARGETS]
    st.dataframe(pd.DataFrame(val_data), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Feature Importance Analysis</div>', unsafe_allow_html=True)
    selected_cqa = st.selectbox("Select CQA", TARGETS)
    imp_df = models[selected_cqa]["importance"].head(10).sort_values("Importance_%", ascending=True)
    
    fig = px.bar(imp_df, x="Importance_%", y="Feature", orientation="h", title=f"Top Factors Affecting {selected_cqa}", color_discrete_sequence=["#0284c7"])
    st.plotly_chart(fig, use_container_width=True)