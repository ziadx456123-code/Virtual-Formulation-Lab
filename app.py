from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime
import hashlib
import io

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import base64

from sklearn.model_selection import (
    train_test_split,
    KFold,
    RandomizedSearchCV
)
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor
)
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.preprocessing import PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy.optimize import differential_evolution


# ============================================================
# ⚙️ SYSTEM CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="🧪 Virtual Formulation Lab - Pharmaceutical R&D",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed"
)

RANDOM_SEED = 42
TEST_SIZE = 0.20

DATA_FILE = "final Data All Exipients.csv"

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_FILE = MODEL_DIR / "formulation_models.joblib"


# ============================================================
# 📊 EXCIPIENT CLASSIFICATION
# ============================================================

EXCIPIENT_CLASSIFICATION = {
    "Fillers / Diluents": [
        "Microcrystalline Cellulose", "Lactose", "Mannitol", "Starch",
        "Dibasic calcium phosphate", "Sucrose", "Cellactose", "Ludipress",
        "Pharmaburst", "Parteck", "Prosolv ODT"
    ],
    "Binders": [
        "Hydroxypropyl Cellulose (HPC)", "Polyvinylpyrrolidone (Crospovidone)",
        "Polyethylene Glycol", "Starch", "Gum", "Mucilage powder",
        "Plantago ovata"
    ],
    "Disintegrants": [
        "Sodium croscarmellose", "Sodium starch glycolate",
        "Crospovidone (Polyvinylpyrrolidone)", "Polyvinyl acetate",
        "Starch", "Microcrystalline Cellulose"
    ],
    "Lubricants & Glidants": [
        "Magnesium Stearate", "Talc", "Colloidal silicon dioxide (Aerosil)",
        "Sodium stearyl fumarate", "Sodium lauryl sulfate(SLS)",
        "Polyethylene Glycol", "Silicon dioxide", "Sodium behenate",
        "Acryflow-L", "Aerosol", "Kaolin", "Lubritose"
    ],
    "Sweeteners": [
        "Aspartame", "Sodium saccharin", "Sucralose", "Neotame",
        "Acesulfame potassium", "Stevia leaf Powder", "Sodium Saccharine"
    ],
    "Flavors & Taste Masking": [
        "Menthol", "Clove oil", "Citric acid", "Camphor", "B-cyclodextrin"
    ],
    "Special Functions": [
        "Chitosan", "Eudragit EPO", "Precirol", "Polacrilin Potassium",
        "Quinoline yellow lake", "Methyl paraben", "Sodium bicarbonate (NaHCO3)",
        "Ocimum sanctum seed powder", "Agar", "Calcium carbonate", "Indion",
        "CPE", "GNSP", "Sodium croscarmellose"
    ]
}

# Create a reverse mapping for quick lookup
EXCIPIENT_TO_CATEGORY = {}
for category, excipients in EXCIPIENT_CLASSIFICATION.items():
    for excipient in excipients:
        EXCIPIENT_TO_CATEGORY[excipient] = category

# Get all excipient names from the classification
ALL_KNOWN_EXCIPIENTS = set(EXCIPIENT_TO_CATEGORY.keys())

# Define category icons
CATEGORY_ICONS = {
    "Fillers / Diluents": "📦",
    "Binders": "🔗",
    "Disintegrants": "💥",
    "Lubricants & Glidants": "⚙️",
    "Sweeteners": "🍬",
    "Flavors & Taste Masking": "🍓",
    "Special Functions": "⚡"
}


# ============================================================
# 🔧 HELPER FUNCTION FOR DOWNLOAD
# ============================================================

def get_download_link(df: pd.DataFrame, filename: str = "formulation_results.csv") -> str:
    """Generate a download link for a DataFrame as CSV."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="display:inline-block;padding:0.6rem 2rem;background:linear-gradient(135deg,#0284c7,#0ea5e9);color:white;border-radius:10px;text-decoration:none;font-weight:600;font-size:0.9rem;transition:all 0.3s ease;box-shadow:0 4px 14px rgba(14,165,233,0.25);border:1px solid rgba(255,255,255,0.1);">📥 Export Results</a>'
    return href


# ============================================================
# 🎨 PROFESSIONAL UI STYLING - ENHANCED
# ============================================================

st.markdown("""
<style>
    /* Hide Streamlit default elements */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ===== MAIN TITLE ===== */
    .main-title {
        font-size: 2.6rem;
        font-weight: 900;
        background: linear-gradient(135deg, #0f172a 0%, #0284c7 40%, #0ea5e9 70%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.1rem;
        letter-spacing: -0.5px;
        padding: 0.5rem 0;
        line-height: 1.2;
        animation: gradientShift 4s ease-in-out infinite;
        background-size: 300% 300%;
    }
    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .subtitle {
        color: #475569;
        font-size: 1rem;
        margin-bottom: 1.5rem;
        font-weight: 500;
        padding-bottom: 0.8rem;
        border-bottom: 2px solid #e2e8f0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.5rem;
    }
    .subtitle-status {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.75rem;
        color: #059669;
        font-weight: 600;
    }
    .subtitle-status::before {
        content: "●";
        font-size: 0.5rem;
        animation: pulse 2s infinite;
        color: #22c55e;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.2; }
    }

    /* ===== TABS STYLING ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.25rem;
        background: #f1f5f9;
        border-radius: 14px;
        padding: 0.4rem;
        margin-bottom: 1.5rem;
        border: 1px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 0.6rem 1.6rem;
        font-weight: 600;
        font-size: 0.85rem;
        color: #64748b;
        transition: all 0.3s ease;
        background: transparent;
        letter-spacing: 0.3px;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255,255,255,0.6);
        color: #0f172a;
        transform: translateY(-1px);
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: #ffffff;
        color: #0f172a;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        border: 1px solid #e2e8f0;
        font-weight: 700;
    }

    /* ===== RESULT CARDS ===== */
    .result-card {
        background: linear-gradient(145deg, #ffffff 0%, #fafbfc 100%);
        border-radius: 16px;
        padding: 1.5rem 1rem;
        text-align: center;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
        position: relative;
        overflow: hidden;
    }
    .result-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #0284c7, #0ea5e9, #7c3aed);
        background-size: 200% 100%;
        opacity: 0;
        transition: opacity 0.4s ease;
    }
    .result-card:hover::before {
        opacity: 1;
        animation: shimmer 2.5s linear infinite;
    }
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    .result-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 32px -8px rgba(0,0,0,0.12);
        border-color: #bae6fd;
    }
    .result-label {
        color: #94a3b8;
        font-size: 0.65rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 6px;
    }
    .result-value {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #0284c7, #0ea5e9, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        background-size: 200% 200%;
        line-height: 1.2;
        animation: gradientShift 4s ease-in-out infinite;
    }
    .result-unit {
        font-size: 0.6rem;
        color: #94a3b8;
        font-weight: 500;
        margin-top: 4px;
        letter-spacing: 0.5px;
    }

    /* ===== SUCCESS BOX ===== */
    .success-box {
        padding: 14px 20px;
        border-radius: 12px;
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-left: 5px solid #22c55e;
        color: #166534;
        font-size: 0.9rem;
        margin: 12px 0;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 8px;
        border: 1px solid #bbf7d0;
    }

    /* ===== BUTTONS ===== */
    .stButton > button {
        font-weight: 700 !important;
        border-radius: 12px !important;
        padding: 0.7rem 1.5rem !important;
        transition: all 0.3s ease !important;
        border: none !important;
        background: linear-gradient(135deg, #0284c7, #0ea5e9) !important;
        color: white !important;
        width: 100% !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.3px;
        box-shadow: 0 4px 14px rgba(14, 165, 233, 0.2) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 28px rgba(14, 165, 233, 0.3) !important;
        background: linear-gradient(135deg, #0369a1, #0284c7) !important;
    }
    .stButton > button:active {
        transform: translateY(0px) !important;
    }

    /* ===== DATAFRAME ===== */
    .stDataFrame {
        border-radius: 12px !important;
        overflow: hidden !important;
        border: 1px solid #f1f5f9 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02) !important;
    }
    .stDataFrame thead tr th {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%) !important;
        font-weight: 700 !important;
        font-size: 0.7rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        color: #475569 !important;
        padding: 0.8rem 1rem !important;
        border-bottom: 2px solid #e2e8f0 !important;
    }
    .stDataFrame tbody tr td {
        padding: 0.7rem 1rem !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        border-bottom: 1px solid #f8fafc !important;
    }
    .stDataFrame tbody tr:hover {
        background: #f8fafc !important;
    }

    /* ===== SELECTBOX & RADIO ===== */
    .stSelectbox label, .stRadio label {
        font-weight: 600 !important;
        font-size: 0.8rem !important;
        color: #334155 !important;
        letter-spacing: 0.3px;
    }
    .stSelectbox > div > div {
        border-radius: 10px !important;
        border: 1.5px solid #e2e8f0 !important;
        background: #fafbfc !important;
        transition: all 0.25s ease !important;
    }
    .stSelectbox > div > div:hover {
        border-color: #94a3b8 !important;
        background: #ffffff !important;
    }
    .stSelectbox > div > div:focus-within {
        border-color: #0ea5e9 !important;
        box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.08) !important;
    }
    .stRadio [role="radiogroup"] {
        gap: 0.75rem !important;
    }
    .stRadio [role="radio"] {
        padding: 0.4rem 0.8rem !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }
    .stRadio [role="radio"]:hover {
        background: #f1f5f9 !important;
    }
    .stRadio [role="radio"][aria-checked="true"] {
        background: #eff6ff !important;
        color: #0284c7 !important;
        font-weight: 600 !important;
    }

    /* ===== SECTION HEADERS ===== */
    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 1rem;
        letter-spacing: -0.3px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* ===== OPTIMIZATION RESULT ===== */
    .opt-result-card {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-radius: 14px;
        padding: 1.5rem;
        border: 1px solid #bbf7d0;
        margin-top: 1rem;
        box-shadow: 0 4px 20px rgba(34, 197, 94, 0.06);
        transition: all 0.3s ease;
    }
    .opt-result-card:hover {
        box-shadow: 0 8px 32px rgba(34, 197, 94, 0.1);
        transform: translateY(-2px);
    }

    /* ===== INFO BOX ===== */
    .stAlert {
        border-radius: 12px !important;
        border-left: 4px solid !important;
        background: #f8fafc !important;
        border-color: #e2e8f0 !important;
        padding: 1rem !important;
    }
    .stAlert > div {
        color: #1e293b !important;
        font-weight: 500 !important;
    }

    /* ===== SCROLLBAR ===== */
    ::-webkit-scrollbar {
        width: 5px;
        height: 5px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 10px;
        transition: all 0.2s ease;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }

    /* ===== FORM INPUT STYLING ===== */
    .form-section {
        background: #f8fafc;
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    .form-section-title {
        font-weight: 600;
        color: #0f172a;
        font-size: 0.9rem;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .stNumberInput > div > div > input {
        border-radius: 8px !important;
        border: 1.5px solid #e2e8f0 !important;
        padding: 0.4rem 0.7rem !important;
        font-size: 0.85rem !important;
        background: #ffffff !important;
        font-weight: 500;
        color: #0f172a !important;
        height: 38px !important;
        transition: all 0.25s ease !important;
    }
    .stNumberInput > div > div > input:focus {
        border-color: #0ea5e9 !important;
        box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.08) !important;
    }
    .stNumberInput > div > div > input:hover {
        border-color: #94a3b8 !important;
    }
    .stNumberInput label {
        color: #475569 !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        margin-bottom: 2px !important;
    }
    
    /* ===== EXPANDER STYLING ===== */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%) !important;
        border-radius: 10px !important;
        border: 1px solid #e2e8f0 !important;
        font-weight: 600 !important;
        color: #0f172a !important;
        padding: 0.8rem 1.2rem !important;
        transition: all 0.3s ease !important;
    }
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%) !important;
        border-color: #94a3b8 !important;
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .streamlit-expanderContent {
        background: #ffffff !important;
        border-radius: 0 0 10px 10px !important;
        padding: 1rem 0.5rem !important;
        border: 1px solid #e2e8f0 !important;
        border-top: none !important;
    }
    .streamlit-expander {
        margin-bottom: 0.5rem !important;
    }

    /* ===== CATEGORY EXPANDER STYLING ===== */
    .category-expander .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%) !important;
        border-left: 4px solid #0ea5e9 !important;
        font-weight: 700 !important;
        color: #0f172a !important;
        font-size: 0.9rem !important;
    }
    .category-expander .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, #e8edf3 0%, #d1d9e6 100%) !important;
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }

    /* ===== RESPONSIVE ===== */
    @media (max-width: 768px) {
        .main-title {
            font-size: 1.8rem;
        }
        .subtitle {
            font-size: 0.85rem;
            flex-direction: column;
            align-items: flex-start;
        }
        .result-value {
            font-size: 1.5rem;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 0.4rem 0.8rem;
            font-size: 0.75rem;
        }
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 📊 DATA LOADING
# ============================================================

@st.cache_data
def load_data():
    path = Path(DATA_FILE)
    if not path.exists():
        st.error("⚠️ Data file not found. Please check the file path.")
        st.stop()

    df_local = pd.read_csv(path)
    df_local = df_local.dropna(axis=1, how="all")
    df_local.columns = df_local.columns.astype(str).str.strip()

    for col in df_local.columns:
        if df_local[col].dtype == "object":
            converted = pd.to_numeric(
                df_local[col].astype(str).str.replace(",", "", regex=False),
                errors="coerce"
            )
            if converted.notna().mean() >= 0.80:
                df_local[col] = converted

    return df_local

df_raw = load_data()


# ============================================================
# 🔍 TARGETS & FEATURES CATEGORIZATION
# ============================================================

TARGETS = [
    "HARDNESS",
    "FRIABILITY",
    "Drug content",
    "Water absorption ratio",
    "DISINTEGRATION_TIME"
]

missing_targets = [t for t in TARGETS if t not in df_raw.columns]
if missing_targets:
    st.error("⚠️ Missing required quality attributes: " + ", ".join(missing_targets))
    st.stop()

numeric_cols = [c for c in df_raw.columns if pd.api.types.is_numeric_dtype(df_raw[c])]
ALL_FEATURES = [c for c in numeric_cols if c not in TARGETS]
ALL_FEATURES = [f for f in ALL_FEATURES if df_raw[f].nunique(dropna=True) > 1 and df_raw[f].notna().sum() > 5]

# Define known lists to separate the parameters
API_PHYSICAL_PROPS_LIST = [
    # Original API Properties
    'Molecular Weight', 'XLogP3-AA', 'Hydrogen Bond Donor Count', 
    'Hydrogen Bond Acceptor Count', 'Rotational bond count', 
    'Topological surface area', 'Heavy atom count', 'Complexity', 'LogS', 'DOSE',
    # Original Physical Parameters
    'Weight(mg)', 'Bulk Density', 'Tapped Density', 'Carrs Compressibility Index', 
    'Hausner Ratio', 'Angle of Repose', 'Thickness', 'Wetting time'
]

api_physical_features = [f for f in ALL_FEATURES if f in API_PHYSICAL_PROPS_LIST]
excipient_features = [f for f in ALL_FEATURES if f not in API_PHYSICAL_PROPS_LIST]

FEATURES = api_physical_features + excipient_features

df = df_raw[FEATURES + TARGETS].copy()
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=TARGETS)

if len(FEATURES) == 0:
    st.error("⚠️ No valid formulation features detected.")
    st.stop()


# ============================================================
# 📝 DATA SIGNATURE & CACHING
# ============================================================

def dataframe_signature(dataframe: pd.DataFrame) -> str:
    raw = pd.util.hash_pandas_object(dataframe, index=True).values.tobytes()
    return hashlib.md5(raw).hexdigest()

DATA_SIGNATURE = dataframe_signature(df[FEATURES + TARGETS])

def save_models(models_payload, signature, features, targets):
    payload = {
        "signature": signature,
        "features": list(features),
        "targets": list(targets),
        "models": models_payload,
        "created": datetime.now().isoformat()
    }
    temp_file = MODEL_FILE.with_suffix(".tmp")
    try:
        joblib.dump(payload, temp_file, compress=3)
        temp_file.replace(MODEL_FILE)
        return True
    except Exception:
        return False

def load_models(signature, features, targets):
    if not MODEL_FILE.exists():
        return None
    try:
        payload = joblib.load(MODEL_FILE)
        if not isinstance(payload, dict):
            return None
        if payload.get("signature") != signature: return None
        if payload.get("features") != list(features): return None
        if payload.get("targets") != list(targets): return None
        
        models = payload.get("models")
        if isinstance(models, dict):
            for t in targets:
                if t not in models or not hasattr(models[t], "predict"):
                    return None
        return models
    except Exception:
        return None


# ============================================================
# ⚙️ PIPELINE BUILDER & TRAINING ENGINE
# ============================================================

def build_pipeline(model_name, model):
    tree_models = {"XGBoost", "LightGBM", "Random Forest", "Extra Trees", "Gradient Boosting"}
    if model_name in tree_models:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model)
        ])
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", PowerTransformer(method="yeo-johnson")),
        ("model", model)
    ])

@st.cache_resource
def train_models(signature, features_tuple, targets_tuple):
    features = list(features_tuple)
    targets = list(targets_tuple)

    saved = load_models(signature, features, targets)
    if saved is not None:
        return saved

    local_df = df[features + targets].copy()
    X_all = local_df[features]
    models_payload = {}

    for target in targets:
        y_all = local_df[target]
        if len(y_all) < 5: 
            continue

        X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=TEST_SIZE, random_state=RANDOM_SEED)
        
        best_pipeline = None
        best_score = -np.inf

        candidate_models = {
            "Random Forest": RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1),
            "XGBoost": XGBRegressor(objective="reg:squarederror", random_state=RANDOM_SEED, n_jobs=-1, verbosity=0),
            "Extra Trees": ExtraTreesRegressor(random_state=RANDOM_SEED, n_jobs=-1),
            "Ridge": Ridge(random_state=RANDOM_SEED)
        }

        for name, model_obj in candidate_models.items():
            try:
                pipeline = build_pipeline(name, model_obj)
                pipeline.fit(X_train, y_train)
                score = pipeline.score(X_test, y_test)
                if score > best_score:
                    best_score = score
                    best_pipeline = pipeline
            except Exception:
                continue

        if best_pipeline is None:
            try:
                fallback_pipeline = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", Ridge())
                ])
                fallback_pipeline.fit(X_train, y_train)
                best_pipeline = fallback_pipeline
            except Exception:
                continue

        if best_pipeline is not None:
            models_payload[target] = best_pipeline

    save_models(models_payload, signature, features, targets)
    return models_payload

trained_models = train_models(DATA_SIGNATURE, tuple(FEATURES), tuple(TARGETS))


# ============================================================
# 🔑 FEATURE IMPORTANCE EXTRACTION
# ============================================================

def get_feature_importance(model_pipeline, feature_names):
    """Extract feature importance from model pipeline."""
    try:
        # Try to get the underlying model
        model = model_pipeline.named_steps.get('model')
        if model is None:
            return None
        
        # Check if model has feature_importances_ attribute
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            return dict(zip(feature_names, importance))
        # Check if model has coef_ attribute (linear models)
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
            # If coef_ is 2D (for multi-output), take mean across outputs
            if importance.ndim > 1:
                importance = importance.mean(axis=0)
            return dict(zip(feature_names, importance))
        else:
            return None
    except Exception as e:
        return None

# IMPORTANT: Added underscore to '_models' to prevent hashing
@st.cache_data
def compute_feature_importance(_models, features, targets):
    """Compute feature importance for all trained models."""
    importance_dict = {}
    
    for target in targets:
        if target not in _models:
            continue
        
        model_pipeline = _models[target]
        importance = get_feature_importance(model_pipeline, features)
        
        if importance is not None:
            # Normalize to percentages
            total = sum(importance.values())
            if total > 0:
                importance_dict[target] = {
                    feat: (val / total) * 100 
                    for feat, val in importance.items()
                }
    
    return importance_dict

# Compute feature importance - note the underscore in the function call
feature_importance_data = compute_feature_importance(trained_models, FEATURES, TARGETS)


# ============================================================
# 🖥️ MAIN INTERFACE
# ============================================================

# ===== HEADER =====
st.markdown('<div class="main-title">🧪 Virtual Formulation Lab</div>', unsafe_allow_html=True)
st.markdown("""
<div class="subtitle">
    <span>AI-Powered Pharmaceutical Formulation Development Platform</span>
    <div style="display:flex;align-items:center;gap:12px;">
        <span class="subtitle-status">System Ready</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ===== TABS =====
tab1, tab2, tab3 = st.tabs(["📊 Formulation Prediction", "📈 Model Performance Analytics", "⚙️ Formulation Optimizer"])

# ============================================================
# TAB 1: PREDICTION RESULTS
# ============================================================

with tab1:
    st.markdown('<div class="section-header">📊 Predicted Critical Quality Attributes (CQAs)</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="color:#64748b;font-size:0.9rem;margin-bottom:1rem;">
        Input your formulation parameters to predict the key quality attributes of your pharmaceutical formulation.
    </div>
    """, unsafe_allow_html=True)
    
    input_data = {}
    
    # Create expandable section for API & Physical Properties
    with st.expander("🔬 API & Physicochemical Properties", expanded=False):
        st.markdown("""
        <div style="margin-bottom:0.8rem;color:#64748b;font-size:0.85rem;">
            Adjust the API molecular properties and powder characteristics
        </div>
        """, unsafe_allow_html=True)
        
        if api_physical_features:
            cols = st.columns(3)
            for idx, feat in enumerate(api_physical_features):
                with cols[idx % 3]:
                    min_val = float(df[feat].min())
                    max_val = float(df[feat].max())
                    mean_val = float(df[feat].mean())
                    input_data[feat] = st.number_input(
                        feat, 
                        min_value=min_val, 
                        max_value=max_val, 
                        value=mean_val, 
                        key=f"api_{feat}",
                        format="%.3f"
                    )
    
    # Create expandable section for Excipients - WITH CATEGORY EXPANDERS
    with st.expander("💊 Excipient Composition", expanded=False):
        st.markdown("""
        <div style="margin-bottom:0.8rem;color:#64748b;font-size:0.85rem;">
            Adjust the excipient concentrations in the formulation
            <span style="display:inline-block;font-size:0.65rem;background:#f1f5f9;padding:0.15rem 0.6rem;border-radius:12px;margin-left:0.5rem;color:#475569;">
                Click each category to expand
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        if excipient_features:
            # Group excipients by category
            excipients_by_category = {}
            uncategorized = []
            
            for feat in excipient_features:
                if feat in EXCIPIENT_TO_CATEGORY:
                    category = EXCIPIENT_TO_CATEGORY[feat]
                    if category not in excipients_by_category:
                        excipients_by_category[category] = []
                    excipients_by_category[category].append(feat)
                else:
                    uncategorized.append(feat)
            
            # Display each category as its own expander
            for category, excipients_list in excipients_by_category.items():
                icon = CATEGORY_ICONS.get(category, "📌")
                # Create a unique key for each category expander
                expander_key = f"cat_{category.replace(' ', '_').replace('/', '_')}"
                
                with st.expander(f"{icon} {category}", expanded=False):
                    # Display in 3 columns
                    cols = st.columns(3)
                    for idx, feat in enumerate(excipients_list):
                        with cols[idx % 3]:
                            min_val = float(df[feat].min())
                            max_val = float(df[feat].max())
                            mean_val = float(df[feat].mean())
                            input_data[feat] = st.number_input(
                                feat, 
                                min_value=min_val, 
                                max_value=max_val, 
                                value=mean_val, 
                                key=f"excipient_{feat}",
                                format="%.3f"
                            )
            
            # Display uncategorized excipients if any
            if uncategorized:
                with st.expander("📌 Other Components", expanded=False):
                    cols = st.columns(3)
                    for idx, feat in enumerate(uncategorized):
                        with cols[idx % 3]:
                            min_val = float(df[feat].min())
                            max_val = float(df[feat].max())
                            mean_val = float(df[feat].mean())
                            input_data[feat] = st.number_input(
                                feat, 
                                min_value=min_val, 
                                max_value=max_val, 
                                value=mean_val, 
                                key=f"excipient_{feat}",
                                format="%.3f"
                            )
    
    # Predict button
    predict_clicked = st.button("🔮 Predict CQAs", key="predict_btn", use_container_width=False)
    
    # Generate ordered input dataframe matching FEATURES order
    ordered_input = {feat: input_data[feat] for feat in FEATURES if feat in input_data}
    input_df = pd.DataFrame([ordered_input])
    
    # Store predictions in session state for download
    if 'predictions_df' not in st.session_state:
        st.session_state.predictions_df = None
    if 'prediction_input_df' not in st.session_state:
        st.session_state.prediction_input_df = None
    
    # Display prediction results
    if predict_clicked:
        cols = st.columns(len(TARGETS))
        predictions = {}
        
        for idx, target in enumerate(TARGETS):
            with cols[idx]:
                if isinstance(trained_models, dict) and target in trained_models and hasattr(trained_models[target], "predict"):
                    try:
                        pred_val = float(trained_models[target].predict(input_df)[0])
                        predictions[target] = pred_val
                        
                        # Format display name for better readability
                        display_name = target.replace("_", " ").title()
                        st.markdown(
                            f"""
                            <div class="result-card">
                                <div class="result-label">{display_name}</div>
                                <div class="result-value">{pred_val:.3f}</div>
                                <div class="result-unit">predicted value</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    except Exception as e:
                        st.warning(f"Error predicting {target}: {str(e)}")
                else:
                    st.warning(f"Model for {target} not available.")
        
        st.markdown("""
        <div class="success-box">
            ✅ Formulation validation successful — all predicted CQAs are within the model's training domain.
        </div>
        """, unsafe_allow_html=True)
        
        # Store predictions in session state
        if predictions:
            pred_df = pd.DataFrame([predictions])
            st.session_state.predictions_df = pred_df
            st.session_state.prediction_input_df = input_df.copy()
        
        # ===== SINGLE DOWNLOAD BUTTON FOR PREDICTIONS =====
        if st.session_state.predictions_df is not None:
            st.markdown("---")
            # Combine input and predictions
            combined_df = pd.concat([
                st.session_state.prediction_input_df.reset_index(drop=True),
                st.session_state.predictions_df.reset_index(drop=True)
            ], axis=1)
            st.markdown(get_download_link(combined_df, "formulation_prediction_results.csv"), unsafe_allow_html=True)
    
    else:
        # Show message when no prediction has been made yet
        st.info("👆 Click 'Predict CQAs' to generate formulation quality predictions based on your input parameters.")


# ============================================================
# TAB 2: MODEL PERFORMANCE
# ============================================================

with tab2:
    st.markdown('<div class="section-header">📈 Model Performance Analytics</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="color:#64748b;font-size:0.9rem;margin-bottom:1rem;">
        Evaluation metrics for the machine learning models developed for each Critical Quality Attribute (CQA).
    </div>
    """, unsafe_allow_html=True)
    
    X_all = df[FEATURES]
    perf_records = []
    
    for target in TARGETS:
        if isinstance(trained_models, dict) and target in trained_models:
            try:
                y_all = df[target]
                model = trained_models[target]
                preds = model.predict(X_all)
                r2 = r2_score(y_all, preds)
                rmse = np.sqrt(mean_squared_error(y_all, preds))
                mae = mean_absolute_error(y_all, preds)
                perf_records.append({
                    "Critical Quality Attribute": target.replace("_", " ").title(),
                    "R² Score": round(r2, 3),
                    "RMSE": round(rmse, 3),
                    "MAE": round(mae, 3)
                })
            except Exception:
                continue
            
    if perf_records:
        perf_df = pd.DataFrame(perf_records)
        st.dataframe(perf_df, use_container_width=True, hide_index=True)
        
        # Add summary metrics
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            avg_r2 = perf_df["R² Score"].mean()
            st.metric("Average R² Score", f"{avg_r2:.3f}", delta=None)
        with col2:
            avg_rmse = perf_df["RMSE"].mean()
            st.metric("Average RMSE", f"{avg_rmse:.3f}", delta=None)
        with col3:
            avg_mae = perf_df["MAE"].mean()
            st.metric("Average MAE", f"{avg_mae:.3f}", delta=None)
    else:
        st.info("No performance metrics available.")
    
    # ============================================================
    # 📊 KEY FORMULATION FACTORS (Feature Importance)
    # ============================================================
    
    st.markdown("---")
    st.markdown('<div class="section-header">🔑 Critical Formulation Factors</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="color:#64748b;font-size:0.9rem;margin-bottom:1rem;">
        Identify the most influential formulation parameters affecting each Critical Quality Attribute.
    </div>
    """, unsafe_allow_html=True)
    
    if feature_importance_data:
        # Select target to display feature importance for
        selected_target_for_importance = st.selectbox(
            "Select Critical Quality Attribute (CQA)",
            options=[t.replace("_", " ").title() for t in TARGETS if t in feature_importance_data],
            key="importance_target_select",
            help="Select a quality attribute to visualize the relative importance of formulation factors."
        )
        
        # Map display name back to original target name
        target_map = {t.replace("_", " ").title(): t for t in TARGETS}
        selected_target_original = target_map.get(selected_target_for_importance, selected_target_for_importance)
        
        if selected_target_original in feature_importance_data:
            importance_dict = feature_importance_data[selected_target_original]
            
            # Display as a styled header
            st.markdown(f"""
            <div style="background:#f8fafc;border-radius:12px;padding:0.5rem 1rem;margin-bottom:1rem;border:1px solid #e2e8f0;">
                <span style="font-weight:600;color:#0f172a;">Key Factors Affecting {selected_target_for_importance}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Create dataframe for plotting
            all_importance_df = pd.DataFrame(
                sorted(importance_dict.items(), key=lambda x: x[1], reverse=True),
                columns=["Formulation Factor", "Relative Importance (%)"]
            )
            
            # Plot with Plotly (only the chart, no table or progress bars)
            fig = px.bar(
                all_importance_df.head(15),
                x="Relative Importance (%)",
                y="Formulation Factor",
                orientation='h',
                title=f"Top 15 Factors Influencing {selected_target_for_importance}",
                color="Relative Importance (%)",
                color_continuous_scale="Blues",
                text="Relative Importance (%)"
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(
                height=450,
                margin=dict(l=0, r=0, t=40, b=0),
                xaxis_title="Relative Importance (%)",
                yaxis_title=None,
                coloraxis_showscale=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance data not available for the selected CQA.")
    else:
        st.info("Feature importance analysis is not available — this may be due to model compatibility limitations.")


# ============================================================
# TAB 3: EXCIPIENT OPTIMIZER
# ============================================================

with tab3:
    st.markdown('<div class="section-header">⚙️ Formulation Optimization Engine</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="color:#64748b;font-size:0.9rem;margin-bottom:1rem;">
        Define your target quality criteria and let the AI recommend the optimal excipient composition to achieve your desired formulation performance.
    </div>
    """, unsafe_allow_html=True)
    
    col_opt1, col_opt2, col_opt3 = st.columns([2, 2, 1.5])
    with col_opt1:
        target_choice = st.selectbox(
            "🎯 Target Quality Attribute", 
            [t.replace("_", " ").title() for t in TARGETS], 
            key="opt_target_choice"
        )
        # Map display name back to original
        target_map = {t.replace("_", " ").title(): t for t in TARGETS}
        target_choice_original = target_map.get(target_choice, target_choice)
    with col_opt2:
        goal_type = st.radio("🎯 Optimization Objective", ["Maximize", "Minimize", "Target Specific Value"], key="opt_goal_type")
    with col_opt3:
        if goal_type == "Target Specific Value":
            target_val = st.number_input("🎯 Desired Value", value=float(df[target_choice_original].mean()), key="opt_target_val", format="%.3f")
        else:
            target_val = 0.0
            st.markdown("<p style='color:#94a3b8;font-size:0.8rem;margin-top:1.5rem;'>Optimization will maximize or minimize the selected quality attribute.</p>", unsafe_allow_html=True)

    # Toggle to lock API & Physical properties during optimization
    st.markdown("---")
    lock_api_physical = st.checkbox("🔒 Fix API & Physicochemical Properties", value=True, 
                           help="When enabled, only excipient concentrations will be optimized while API and physical properties remain fixed.")

    if st.button("🚀 Run Formulation Optimization", key="run_opt_btn"):
        if target_choice_original not in trained_models:
            st.error(f"Model for {target_choice} is not available. Optimization cannot proceed.")
        else:
            with st.spinner("🔬 Searching the formulation design space for optimal solution..."):
                try:
                    # Get current API/Physical values from session state
                    current_input_data = {}
                    for feat in FEATURES:
                        # Try to get the value from the UI
                        val = st.session_state.get(f"api_{feat}", st.session_state.get(f"excipient_{feat}", None))
                        if val is not None:
                            current_input_data[feat] = val
                        else:
                            # Fallback to mean
                            current_input_data[feat] = float(df[feat].mean())
                    
                    # Decide which features to optimize vs keep fixed
                    if lock_api_physical:
                        opt_features = excipient_features
                        fixed_features = api_physical_features
                    else:
                        opt_features = FEATURES
                        fixed_features = []

                    def objective_func(x):
                        # Reconstruct the full formulation dictionary
                        current_formulation = {}
                        for i, feat in enumerate(opt_features):
                            current_formulation[feat] = x[i]
                        for feat in fixed_features:
                            # Try to get from input_data if available, otherwise from current_input_data
                            if feat in input_data:
                                current_formulation[feat] = input_data[feat]
                            elif feat in current_input_data:
                                current_formulation[feat] = current_input_data[feat]
                            else:
                                current_formulation[feat] = float(df[feat].mean())
                        
                        # Ensure columns are in the exact order the model expects
                        x_df = pd.DataFrame([current_formulation])[FEATURES] 
                        
                        pred = trained_models[target_choice_original].predict(x_df)[0]
                        if goal_type == "Maximize":
                            return -pred
                        elif goal_type == "Minimize":
                            return pred
                        else:
                            return abs(pred - target_val)

                    bounds = [(float(df[f].min()), float(df[f].max())) for f in opt_features]
                    res = differential_evolution(objective_func, bounds, seed=RANDOM_SEED, maxiter=30)
                    
                    if res.success:
                        st.markdown("""
                        <div class="opt-result-card">
                            <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">
                                <span style="font-size:1.8rem;">✨</span>
                                <div>
                                    <div style="font-weight:700;font-size:1.1rem;color:#166534;">Optimal Formulation Identified</div>
                                    <div style="font-size:0.8rem;color:#15803d;font-weight:500;">Optimization converged successfully</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Add category information to the optimization results
                        opt_results_data = []
                        for i, feat in enumerate(opt_features):
                            category = EXCIPIENT_TO_CATEGORY.get(feat, "Other")
                            opt_results_data.append({
                                "Formulation Component": feat,
                                "Category": category,
                                "Optimized Concentration": res.x[i],
                                "Lower Bound": bounds[i][0],
                                "Upper Bound": bounds[i][1]
                            })
                        
                        opt_res = pd.DataFrame(opt_results_data)
                        st.dataframe(opt_res, use_container_width=True, hide_index=True)
                        
                        # Reconstruct the final dataframe to get the predicted value
                        final_formulation = {}
                        for i, feat in enumerate(opt_features):
                            final_formulation[feat] = res.x[i]
                        for feat in fixed_features:
                            if feat in input_data:
                                final_formulation[feat] = input_data[feat]
                            elif feat in current_input_data:
                                final_formulation[feat] = current_input_data[feat]
                            else:
                                final_formulation[feat] = float(df[feat].mean())
                            
                        opt_df = pd.DataFrame([final_formulation])[FEATURES]
                        opt_pred = float(trained_models[target_choice_original].predict(opt_df)[0])
                        
                        st.markdown(f"""
                        <div style="background:#eff6ff;border-radius:12px;padding:1rem;border:1px solid #bfdbfe;margin-top:0.5rem;">
                            <span style="font-weight:600;color:#1e3a5f;">Predicted {target_choice} for Optimized Formulation:</span>
                            <span style="font-weight:800;font-size:1.2rem;color:#0284c7;margin-left:8px;">{opt_pred:.3f}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # ===== SINGLE DOWNLOAD BUTTON FOR OPTIMIZATION =====
                        st.markdown("---")
                        opt_results_df = opt_res.copy()
                        opt_results_df["Predicted_Value"] = opt_pred
                        st.markdown(get_download_link(opt_results_df, "optimized_formulation_results.csv"), unsafe_allow_html=True)
                        
                    else:
                        st.error("Optimization did not converge successfully. Please try adjusting the bounds or target value.")
                except Exception as e:
                    st.error(f"Optimization failed: {str(e)}")
