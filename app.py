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
    page_title="🧪 The Formula | Ziad Ibrahim",
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
# 👨‍🔬 DEVELOPER BRANDING
# ============================================================

DEVELOPER_NAME = "Ziad Ibrahim"


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

EXCIPIENT_TO_CATEGORY = {}
for category, excipients in EXCIPIENT_CLASSIFICATION.items():
    for excipient in excipients:
        EXCIPIENT_TO_CATEGORY[excipient] = category

ALL_KNOWN_EXCIPIENTS = set(EXCIPIENT_TO_CATEGORY.keys())

CATEGORY_ICONS = {
    "Fillers / Diluents": "📦",
    "Binders": "🔗",
    "Disintegrants": "💥",
    "Lubricants & Glidants": "⚙️",
    "Sweeteners": "🍬",
    "Flavors & Taste Masking": "🍓",
    "Special Functions": "⚡"
}

CATEGORY_COLORS = {
    "Fillers / Diluents": "#3B82F6",
    "Binders": "#8B5CF6",
    "Disintegrants": "#EF4444",
    "Lubricants & Glidants": "#F59E0B",
    "Sweeteners": "#EC4899",
    "Flavors & Taste Masking": "#10B981",
    "Special Functions": "#6366F1"
}


# ============================================================
# 🔧 HELPER FUNCTION FOR DOWNLOAD
# ============================================================

def get_download_link(df: pd.DataFrame, filename: str = "formulation_results.csv") -> str:
    if not filename.endswith(".csv"):
        filename += ".csv"
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="display:inline-block;padding:0.7rem 2.2rem;background:linear-gradient(135deg,#1a56db,#3b82f6);color:white;border-radius:12px;text-decoration:none;font-weight:600;font-size:0.95rem;transition:all 0.3s ease;box-shadow:0 4px 16px rgba(59,130,246,0.3);border:1px solid rgba(255,255,255,0.15);letter-spacing:0.3px;">📥 Download "{filename}"</a>'
    return href


# ============================================================
# 🎨 PROFESSIONAL UI STYLING
# ============================================================

st.markdown("""
<style>
    #MainMenu, footer, header { visibility: hidden; }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ===== HEADER ROW (TITLE LEFT + BADGE RIGHT) ===== */
    .header-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        flex-wrap: wrap;
        margin-bottom: 0.5rem;
    }
    .header-row .main-title {
        margin-bottom: 0;
        padding: 0.2rem 0;
    }

    /* ===== MAIN TITLE ===== */
    .main-title {
        font-size: 2.8rem;
        font-weight: 900;
        background: linear-gradient(135deg, #0f172a 0%, #1a56db 30%, #3b82f6 60%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.4rem;
        letter-spacing: -0.5px;
        padding: 0.5rem 0;
        line-height: 1.2;
        animation: gradientShift 5s ease-in-out infinite;
        background-size: 300% 300%;
    }
    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    /* ===== SUBTITLE ===== */
    .subtitle {
        color: var(--text-color-secondary, #475569);
        font-size: 1rem;
        margin-bottom: 1.5rem;
        font-weight: 500;
        padding-bottom: 0.8rem;
        border-bottom: 2px solid var(--border-color, #e2e8f0);
        display: flex;
        align-items: center;
        gap: 0.5rem;
        letter-spacing: 0.1px;
    }

    /* ===== DEVELOPER BADGE (SMALL - INLINE RIGHT) ===== */
    .dev-badge {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        background: linear-gradient(135deg, rgba(26,86,219,0.07), rgba(139,92,246,0.07));
        border: 1px solid rgba(59,130,246,0.2);
        border-radius: 999px;
        padding: 0.32rem 0.95rem 0.32rem 0.35rem;
        backdrop-filter: blur(6px);
        box-shadow: 0 1px 6px rgba(59,130,246,0.06);
        flex-shrink: 0;
        white-space: nowrap;
    }
    .dev-badge-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 24px;
        height: 24px;
        border-radius: 50%;
        background: linear-gradient(135deg, #1a56db, #8b5cf6);
        color: white;
        font-size: 0.72rem;
        box-shadow: 0 2px 6px rgba(59,130,246,0.25);
        flex-shrink: 0;
    }
    .dev-badge-name {
        font-weight: 700;
        font-size: 0.78rem;
        color: var(--text-color-primary, #0f172a);
        letter-spacing: 0.2px;
        line-height: 1;
    }

    /* ===== TABS STYLING ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.25rem;
        background: var(--bg-secondary, #f1f5f9);
        border-radius: 16px;
        padding: 0.4rem;
        margin-bottom: 1.5rem;
        border: 1px solid var(--border-color, #e2e8f0);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        padding: 0.7rem 2rem;
        font-weight: 600;
        font-size: 0.9rem;
        color: var(--text-color-secondary, #64748b);
        transition: all 0.3s ease;
        background: transparent;
        letter-spacing: 0.3px;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: var(--bg-hover, rgba(255,255,255,0.7));
        color: var(--text-color-primary, #0f172a);
        transform: translateY(-1px);
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: var(--bg-card, #ffffff);
        color: #1a56db;
        box-shadow: 0 4px 16px var(--shadow-color, rgba(0,0,0,0.08));
        border: 1px solid #dbeafe;
        font-weight: 700;
    }

    /* ===== RESULT CARDS ===== */
    .result-card {
        background: var(--bg-card, linear-gradient(145deg, #ffffff 0%, #f8fafc 100%));
        border-radius: 18px;
        padding: 1.8rem 1rem;
        text-align: center;
        border: 1px solid var(--border-color, #e2e8f0);
        box-shadow: 0 2px 12px var(--shadow-color, rgba(0,0,0,0.02));
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
        background: linear-gradient(90deg, #1a56db, #3b82f6, #8b5cf6);
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
        transform: translateY(-6px);
        box-shadow: 0 16px 48px -12px var(--shadow-color, rgba(0,0,0,0.15));
        border-color: #93c5fd;
    }
    .result-label {
        color: var(--text-color-muted, #94a3b8);
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }
    .result-value {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1a56db, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        background-size: 200% 200%;
        line-height: 1.2;
        animation: gradientShift 4s ease-in-out infinite;
    }
    .result-unit {
        font-size: 0.65rem;
        color: var(--text-color-muted, #94a3b8);
        font-weight: 500;
        margin-top: 4px;
        letter-spacing: 0.5px;
    }

    /* ===== SUCCESS BOX ===== */
    .success-box {
        padding: 16px 24px;
        border-radius: 14px;
        background: var(--success-bg, linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%));
        border-left: 6px solid #22c55e;
        color: var(--success-text, #166534);
        font-size: 0.95rem;
        margin: 16px 0;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 10px;
        border: 1px solid #bbf7d0;
        box-shadow: 0 2px 12px rgba(34,197,94,0.08);
    }

    /* ===== BUTTONS ===== */
    .stButton > button {
        font-weight: 700 !important;
        border-radius: 14px !important;
        padding: 0.8rem 2rem !important;
        transition: all 0.3s ease !important;
        border: none !important;
        background: linear-gradient(135deg, #1a56db, #3b82f6) !important;
        color: white !important;
        width: auto !important;
        font-size: 1rem !important;
        letter-spacing: 0.3px;
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.25) !important;
    }
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 32px rgba(59, 130, 246, 0.35) !important;
        background: linear-gradient(135deg, #1e40af, #2563eb) !important;
    }
    .stButton > button:active {
        transform: translateY(0px) !important;
    }

    /* ===== DATAFRAME ===== */
    .stDataFrame {
        border-radius: 14px !important;
        overflow: hidden !important;
        border: 1px solid var(--border-color, #f1f5f9) !important;
        box-shadow: 0 2px 12px var(--shadow-color, rgba(0,0,0,0.02)) !important;
    }
    .stDataFrame thead tr th {
        background: var(--bg-secondary, linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%)) !important;
        font-weight: 700 !important;
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        color: var(--text-color-primary, #334155) !important;
        padding: 0.8rem 1rem !important;
        border-bottom: 2px solid var(--border-color, #cbd5e1) !important;
    }
    .stDataFrame tbody tr td {
        padding: 0.7rem 1rem !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        border-bottom: 1px solid var(--border-color, #f8fafc) !important;
        color: var(--text-color-primary, #1e293b) !important;
    }
    .stDataFrame tbody tr:hover {
        background: var(--bg-hover, #f1f5f9) !important;
    }

    /* ===== SELECTBOX & RADIO ===== */
    .stSelectbox label, .stRadio label {
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        color: var(--text-color-primary, #1e293b) !important;
        letter-spacing: 0.3px;
        margin-bottom: 4px !important;
    }
    .stSelectbox > div > div {
        border-radius: 12px !important;
        border: 2px solid var(--border-color, #e2e8f0) !important;
        background: var(--bg-card, #ffffff) !important;
        transition: all 0.25s ease !important;
    }
    .stSelectbox > div > div:hover {
        border-color: var(--text-color-muted, #94a3b8) !important;
        box-shadow: 0 2px 8px var(--shadow-color, rgba(0,0,0,0.02));
    }
    .stSelectbox > div > div:focus-within {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.08) !important;
    }
    .stRadio [role="radiogroup"] {
        gap: 0.75rem !important;
        background: var(--bg-secondary, #f8fafc);
        padding: 0.5rem;
        border-radius: 12px;
        border: 1px solid var(--border-color, #e2e8f0);
    }
    .stRadio [role="radio"] {
        padding: 0.5rem 1.2rem !important;
        border-radius: 10px !important;
        transition: all 0.2s ease !important;
        font-weight: 500 !important;
        color: var(--text-color-primary, #1e293b) !important;
    }
    .stRadio [role="radio"]:hover {
        background: var(--bg-hover, #e2e8f0) !important;
    }
    .stRadio [role="radio"][aria-checked="true"] {
        background: #dbeafe !important;
        color: #1a56db !important;
        font-weight: 700 !important;
        border: 1px solid #93c5fd;
    }

    /* ===== SECTION HEADERS ===== */
    .section-header {
        font-size: 1.3rem;
        font-weight: 800;
        color: var(--text-color-primary, #0f172a);
        margin-bottom: 1rem;
        letter-spacing: -0.3px;
        display: flex;
        align-items: center;
        gap: 10px;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid var(--border-color, #e2e8f0);
    }
    .section-header-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: var(--bg-highlight, linear-gradient(135deg, #dbeafe, #eff6ff));
        border-radius: 10px;
        padding: 0.3rem 0.6rem;
        font-size: 1.1rem;
    }

    /* ===== OPTIMIZATION RESULT ===== */
    .opt-result-card {
        background: var(--success-bg, linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%));
        border-radius: 16px;
        padding: 1.8rem;
        border: 1px solid #bbf7d0;
        margin-top: 1rem;
        box-shadow: 0 4px 24px rgba(34, 197, 94, 0.08);
        transition: all 0.3s ease;
    }
    .opt-result-card:hover {
        box-shadow: 0 8px 40px rgba(34, 197, 94, 0.12);
        transform: translateY(-2px);
    }
    .opt-result-card div {
        color: var(--success-text, #166534) !important;
    }

    /* ===== INFO BOX ===== */
    .stAlert {
        border-radius: 14px !important;
        border-left: 5px solid !important;
        background: var(--bg-secondary, #f8fafc) !important;
        border-color: var(--border-color, #e2e8f0) !important;
        padding: 1.2rem !important;
    }
    .stAlert > div {
        color: var(--text-color-primary, #1e293b) !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }

    /* ===== SCROLLBAR ===== */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: var(--bg-secondary, #f1f5f9);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background: var(--text-color-muted, #94a3b8);
        border-radius: 10px;
        transition: all 0.2s ease;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-color-secondary, #64748b);
    }

    /* ===== FORM INPUT STYLING ===== */
    .stNumberInput > div > div > input {
        border-radius: 10px !important;
        border: 2px solid var(--border-color, #e2e8f0) !important;
        padding: 0.5rem 0.8rem !important;
        font-size: 0.85rem !important;
        background: var(--bg-card, #ffffff) !important;
        font-weight: 500;
        color: var(--text-color-primary, #0f172a) !important;
        height: 42px !important;
        transition: all 0.25s ease !important;
    }
    .stNumberInput > div > div > input:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.08) !important;
    }
    .stNumberInput > div > div > input:hover {
        border-color: var(--text-color-muted, #94a3b8) !important;
    }
    .stNumberInput label {
        color: var(--text-color-secondary, #475569) !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        margin-bottom: 4px !important;
    }
    
    /* ===== EXPANDER STYLING ===== */
    .streamlit-expanderHeader {
        background: var(--bg-secondary, linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%)) !important;
        border-radius: 12px !important;
        border: 1px solid var(--border-color, #e2e8f0) !important;
        font-weight: 700 !important;
        color: var(--text-color-primary, #0f172a) !important;
        padding: 0.9rem 1.4rem !important;
        transition: all 0.3s ease !important;
        font-size: 0.9rem !important;
    }
    .streamlit-expanderHeader:hover {
        background: var(--bg-hover, linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%)) !important;
        border-color: var(--text-color-muted, #94a3b8) !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 16px var(--shadow-color, rgba(0,0,0,0.04));
    }
    .streamlit-expanderContent {
        background: var(--bg-card, #ffffff) !important;
        border-radius: 0 0 12px 12px !important;
        padding: 1.2rem 0.8rem !important;
        border: 1px solid var(--border-color, #e2e8f0) !important;
        border-top: none !important;
    }
    .streamlit-expander {
        margin-bottom: 0.5rem !important;
    }

    /* ===== METRIC CARDS ===== */
    .metric-card {
        background: var(--bg-card, #ffffff);
        border-radius: 14px;
        padding: 1.2rem 1rem;
        border: 1px solid var(--border-color, #e2e8f0);
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px var(--shadow-color, rgba(0,0,0,0.02));
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px var(--shadow-color, rgba(0,0,0,0.06));
        border-color: #93c5fd;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #1a56db;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 0.75rem;
        color: var(--text-color-muted, #94a3b8);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }

    /* ===== INFO TEXT ===== */
    .info-text {
        color: var(--text-color-secondary, #64748b);
        font-size: 0.95rem;
        margin-bottom: 1.2rem;
        background: var(--bg-secondary, #f8fafc);
        padding: 0.8rem 1.2rem;
        border-radius: 12px;
        border: 1px solid var(--border-color, #e2e8f0);
    }
    .info-box {
        color: var(--text-color-secondary, #64748b);
        font-size: 0.85rem;
        background: var(--bg-secondary, #f1f5f9);
        padding: 0.3rem 0.8rem;
        border-radius: 8px;
    }
    .prediction-box {
        background: var(--bg-highlight, #eff6ff);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        border: 1px solid #bfdbfe;
        margin-top: 0.8rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.5rem;
    }
    .prediction-box span:first-child {
        font-weight: 700;
        color: var(--text-color-primary, #1e3a5f);
        font-size: 0.95rem;
    }
    .prediction-box span:last-child {
        font-weight: 800;
        font-size: 1.4rem;
        color: #1a56db;
    }

    /* ===== RESPONSIVE ===== */
    @media (max-width: 768px) {
        .header-row {
            flex-direction: column;
            align-items: flex-start;
            gap: 10px;
        }
        .main-title {
            font-size: 1.8rem;
        }
        .subtitle {
            font-size: 0.85rem;
        }
        .result-value {
            font-size: 1.6rem;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 0.4rem 1rem;
            font-size: 0.75rem;
        }
        .stButton > button {
            width: 100% !important;
        }
    }

    /* ===== DARK MODE OVERRIDES ===== */
    @media (prefers-color-scheme: dark) {
        :root {
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-card: #1e293b;
            --bg-hover: #334155;
            --bg-highlight: #1e293b;
            --text-color-primary: #f1f5f9;
            --text-color-secondary: #94a3b8;
            --text-color-muted: #64748b;
            --border-color: #334155;
            --shadow-color: rgba(0,0,0,0.4);
            --success-bg: #1a2e1a;
            --success-text: #86efac;
        }
        .dev-badge {
            background: linear-gradient(135deg, rgba(59,130,246,0.14), rgba(139,92,246,0.14)) !important;
            border-color: rgba(59,130,246,0.3) !important;
        }
        .dev-badge-name {
            color: #f1f5f9 !important;
        }
        .result-card {
            background: var(--bg-card) !important;
            border-color: var(--border-color) !important;
        }
        .result-card:hover {
            border-color: #3b82f6 !important;
        }
        .stDataFrame thead tr th {
            background: var(--bg-secondary) !important;
            color: #f1f5f9 !important;
        }
        .stDataFrame tbody tr td {
            color: #e2e8f0 !important;
            border-color: var(--border-color) !important;
        }
        .stDataFrame tbody tr:hover {
            background: var(--bg-hover) !important;
        }
        .stSelectbox > div > div {
            background: var(--bg-card) !important;
            border-color: var(--border-color) !important;
        }
        .stRadio [role="radiogroup"] {
            background: var(--bg-secondary) !important;
            border-color: var(--border-color) !important;
        }
        .stRadio [role="radio"] {
            color: #e2e8f0 !important;
        }
        .stRadio [role="radio"]:hover {
            background: var(--bg-hover) !important;
        }
        .stRadio [role="radio"][aria-checked="true"] {
            background: #1e3a5f !important;
            color: #60a5fa !important;
            border-color: #3b82f6 !important;
        }
        .streamlit-expanderHeader {
            background: var(--bg-secondary) !important;
            border-color: var(--border-color) !important;
            color: #f1f5f9 !important;
        }
        .streamlit-expanderHeader:hover {
            background: var(--bg-hover) !important;
        }
        .streamlit-expanderContent {
            background: var(--bg-card) !important;
            border-color: var(--border-color) !important;
        }
        .stNumberInput > div > div > input {
            background: var(--bg-card) !important;
            border-color: var(--border-color) !important;
            color: #f1f5f9 !important;
        }
        .stNumberInput > div > div > input:hover {
            border-color: #64748b !important;
        }
        .stNumberInput label {
            color: #94a3b8 !important;
        }
        .section-header {
            color: #f1f5f9 !important;
            border-bottom-color: var(--border-color) !important;
        }
        .section-header-icon {
            background: var(--bg-secondary) !important;
        }
        .subtitle {
            color: #94a3b8 !important;
            border-bottom-color: var(--border-color) !important;
        }
        .success-box {
            background: #1a2e1a !important;
            border-color: #166534 !important;
            color: #86efac !important;
        }
        .opt-result-card {
            background: #1a2e1a !important;
            border-color: #166534 !important;
        }
        .opt-result-card div {
            color: #86efac !important;
        }
        .metric-card {
            background: var(--bg-card) !important;
            border-color: var(--border-color) !important;
        }
        .metric-label {
            color: #64748b !important;
        }
        .info-text {
            background: var(--bg-secondary) !important;
            border-color: var(--border-color) !important;
            color: #94a3b8 !important;
        }
        .info-box {
            background: var(--bg-secondary) !important;
            color: #94a3b8 !important;
        }
        .prediction-box {
            background: #1a2e3a !important;
            border-color: #1e40af !important;
        }
        .prediction-box span:first-child {
            color: #93c5fd !important;
        }
        .stAlert {
            background: var(--bg-secondary) !important;
            border-color: var(--border-color) !important;
        }
        .stAlert > div {
            color: #e2e8f0 !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            background: var(--bg-secondary) !important;
            border-color: var(--border-color) !important;
        }
        .stTabs [data-baseweb="tab"] {
            color: #94a3b8 !important;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background: var(--bg-hover) !important;
            color: #f1f5f9 !important;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: var(--bg-card) !important;
            color: #60a5fa !important;
            border-color: #3b82f6 !important;
        }
        .main-title {
            background: linear-gradient(135deg, #f1f5f9 0%, #60a5fa 30%, #93c5fd 60%, #a78bfa 100%) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            background-clip: text !important;
        }
        ::-webkit-scrollbar-track {
            background: var(--bg-secondary) !important;
        }
        ::-webkit-scrollbar-thumb {
            background: #475569 !important;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #64748b !important;
        }
    }

    /* ===== LIGHT MODE OVERRIDES ===== */
    @media (prefers-color-scheme: light) {
        :root {
            --bg-primary: #ffffff;
            --bg-secondary: #f1f5f9;
            --bg-card: #ffffff;
            --bg-hover: #f1f5f9;
            --bg-highlight: #eff6ff;
            --text-color-primary: #0f172a;
            --text-color-secondary: #475569;
            --text-color-muted: #94a3b8;
            --border-color: #e2e8f0;
            --shadow-color: rgba(0,0,0,0.08);
            --success-bg: #f0fdf4;
            --success-text: #166534;
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

TARGET_DISPLAY_NAMES = {
    "HARDNESS": "Hardness (N)",
    "FRIABILITY": "Friability (%)",
    "Drug content": "Drug Content (%)",
    "Water absorption ratio": "Water Absorption Ratio",
    "DISINTEGRATION_TIME": "Disintegration Time (sec)"
}

missing_targets = [t for t in TARGETS if t not in df_raw.columns]
if missing_targets:
    st.error("⚠️ Missing required quality attributes: " + ", ".join(missing_targets))
    st.stop()

numeric_cols = [c for c in df_raw.columns if pd.api.types.is_numeric_dtype(df_raw[c])]
ALL_FEATURES = [c for c in numeric_cols if c not in TARGETS]
ALL_FEATURES = [f for f in ALL_FEATURES if df_raw[f].nunique(dropna=True) > 1 and df_raw[f].notna().sum() > 5]

API_PHYSICAL_PROPS_LIST = [
    'Molecular Weight', 'XLogP3-AA', 'Hydrogen Bond Donor Count', 
    'Hydrogen Bond Acceptor Count', 'Rotational bond count', 
    'Topological surface area', 'Heavy atom count', 'Complexity', 'LogS', 'DOSE',
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
    try:
        model = model_pipeline.named_steps.get('model')
        if model is None:
            return None
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            return dict(zip(feature_names, importance))
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
            if importance.ndim > 1:
                importance = importance.mean(axis=0)
            return dict(zip(feature_names, importance))
        else:
            return None
    except Exception as e:
        return None

@st.cache_data
def compute_feature_importance(_models, features, targets):
    importance_dict = {}
    
    for target in targets:
        if target not in _models:
            continue
        
        model_pipeline = _models[target]
        importance = get_feature_importance(model_pipeline, features)
        
        if importance is not None:
            total = sum(importance.values())
            if total > 0:
                importance_dict[target] = {
                    feat: (val / total) * 100 
                    for feat, val in importance.items()
                }
    
    return importance_dict

feature_importance_data = compute_feature_importance(trained_models, FEATURES, TARGETS)


# ============================================================
# 🖥️ MAIN INTERFACE
# ============================================================

# ===== HEADER (TITLE LEFT + BADGE RIGHT) =====
st.markdown(f"""
<div class="header-row">
    <div class="main-title">🧪 The Formula — AI Formulation Lab</div>
    <div class="dev-badge">
        <div class="dev-badge-icon">👨‍🔬</div>
        <span class="dev-badge-name">{DEVELOPER_NAME}</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="subtitle">
    <span>AI-Powered Pharmaceutical Formulation Development Platform</span>
</div>
""", unsafe_allow_html=True)

# ===== TABS =====
tab1, tab2, tab3 = st.tabs([
    "📊 Formulation Prediction",
    "⚙️ Formulation Optimizer",
    "📈 Model Performance Analytics"
])

# ============================================================
# TAB 1: PREDICTION RESULTS
# ============================================================

with tab1:
    st.markdown("""
    <div class="section-header">
        <span class="section-header-icon">📊</span>
        Predicted Critical Quality Attributes (CQAs)
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="info-text">
        💡 Input your formulation parameters below and click <strong>"🔮 Predict CQAs"</strong> to generate predicted quality attributes.
    </div>
    """, unsafe_allow_html=True)
    
    input_data = {}
    
    with st.expander("🔬 API & Physicochemical Properties", expanded=True):
        st.markdown("""
        <div class="info-box">
            ⚙️ Adjust the API molecular properties and powder characteristics
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
    
    with st.expander("💊 Excipient Composition", expanded=True):
        st.markdown("""
        <div class="info-box">
            📌 Adjust the excipient concentrations — click each category to expand
        </div>
        """, unsafe_allow_html=True)
        
        if excipient_features:
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
            
            for category, excipients_list in excipients_by_category.items():
                icon = CATEGORY_ICONS.get(category, "📌")
                color = CATEGORY_COLORS.get(category, "#3B82F6")
                expander_key = f"cat_{category.replace(' ', '_').replace('/', '_')}"
                
                with st.expander(f"{icon} {category}", expanded=False):
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
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_clicked = st.button("🔮 Predict CQAs", key="predict_btn", use_container_width=True)
    
    ordered_input = {feat: input_data[feat] for feat in FEATURES if feat in input_data}
    input_df = pd.DataFrame([ordered_input])
    
    if 'predictions_df' not in st.session_state:
        st.session_state.predictions_df = None
    if 'prediction_input_df' not in st.session_state:
        st.session_state.prediction_input_df = None
    
    if predict_clicked:
        cols = st.columns(len(TARGETS))
        predictions = {}
        
        for idx, target in enumerate(TARGETS):
            with cols[idx]:
                if isinstance(trained_models, dict) and target in trained_models and hasattr(trained_models[target], "predict"):
                    try:
                        pred_val = float(trained_models[target].predict(input_df)[0])
                        predictions[target] = pred_val
                        
                        display_name = TARGET_DISPLAY_NAMES.get(target, target.replace("_", " ").title())
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
        
        if predictions:
            pred_df = pd.DataFrame([predictions])
            st.session_state.predictions_df = pred_df
            st.session_state.prediction_input_df = input_df.copy()
        
        if st.session_state.predictions_df is not None:
            st.markdown("---")
            st.markdown("""
            <div style="font-weight:700;font-size:1.1rem;color:var(--text-color-primary, #0f172a);margin-bottom:0.5rem;">
                💾 Export Prediction Results
            </div>
            """, unsafe_allow_html=True)
            
            col_fn1, col_fn2 = st.columns([2, 1])
            with col_fn1:
                custom_filename_pred = st.text_input("File name:", value="the_formula_prediction_results", key="custom_filename_pred")
            with col_fn2:
                combined_df = pd.concat([
                    st.session_state.prediction_input_df.reset_index(drop=True),
                    st.session_state.predictions_df.reset_index(drop=True)
                ], axis=1)
                st.markdown(get_download_link(combined_df, custom_filename_pred), unsafe_allow_html=True)
    
    else:
        st.info("👆 Click 'Predict CQAs' to generate formulation quality predictions based on your input parameters.")


# ============================================================
# TAB 2: EXCIPIENT OPTIMIZER
# ============================================================

with tab2:
    st.markdown("""
    <div class="section-header">
        <span class="section-header-icon">⚙️</span>
        Formulation Optimization Engine
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="info-text">
        🎯 Define your target quality criteria and let the AI recommend the optimal excipient composition.
    </div>
    """, unsafe_allow_html=True)
    
    col_opt1, col_opt2, col_opt3 = st.columns([2, 2, 1.5])
    with col_opt1:
        target_display_options = [TARGET_DISPLAY_NAMES.get(t, t.replace("_", " ").title()) for t in TARGETS]
        target_choice_display = st.selectbox(
            "🎯 Target Quality Attribute", 
            target_display_options, 
            key="opt_target_choice"
        )
        reverse_display_map = {v: k for k, v in TARGET_DISPLAY_NAMES.items()}
        target_choice = reverse_display_map.get(target_choice_display, target_choice_display)
    with col_opt2:
        goal_type = st.radio("🎯 Optimization Objective", ["Maximize", "Minimize", "Target Specific Value"], key="opt_goal_type")
    with col_opt3:
        if goal_type == "Target Specific Value":
            target_val = st.number_input("🎯 Desired Value", value=float(df[target_choice].mean()), key="opt_target_val", format="%.3f")
        else:
            target_val = 0.0
            st.markdown(f"""
            <div style="background:var(--bg-secondary, #f1f5f9);border-radius:10px;padding:0.8rem;margin-top:1.5rem;border:1px solid var(--border-color, #e2e8f0);color:var(--text-color-secondary, #475569);font-size:0.85rem;text-align:center;">
                ⚡ Optimization will <strong>{goal_type.lower()}</strong> the selected quality attribute.
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    lock_api_physical = st.checkbox("🔒 Fix API & Physicochemical Properties", value=True, 
                           help="When enabled, only excipient concentrations will be optimized while API and physical properties remain fixed.")

    col_opt_btn1, col_opt_btn2, col_opt_btn3 = st.columns([1, 2, 1])
    with col_opt_btn2:
        optimize_clicked = st.button("🚀 Run Formulation Optimization", key="run_opt_btn", use_container_width=True)

    if optimize_clicked:
        if target_choice not in trained_models:
            st.error(f"❌ Model for {target_choice_display} is not available. Optimization cannot proceed.")
        else:
            with st.spinner("🔬 Searching the formulation design space for optimal solution..."):
                try:
                    base_formulation = {}
                    
                    for feat in api_physical_features:
                        if feat in input_data:
                            base_formulation[feat] = input_data[feat]
                        else:
                            session_val = st.session_state.get(f"api_{feat}", None)
                            if session_val is not None:
                                base_formulation[feat] = session_val
                            else:
                                base_formulation[feat] = float(df[feat].mean())
                    
                    for feat in excipient_features:
                        session_val = st.session_state.get(f"excipient_{feat}", None)
                        if session_val is not None:
                            base_formulation[feat] = session_val
                        elif feat in input_data:
                            base_formulation[feat] = input_data[feat]
                        else:
                            base_formulation[feat] = float(df[feat].mean())
                    
                    if lock_api_physical:
                        opt_features = excipient_features
                    else:
                        opt_features = FEATURES

                    for feat in FEATURES:
                        if feat not in base_formulation:
                            base_formulation[feat] = float(df[feat].mean())
                    
                    fixed_values = {}
                    for feat in FEATURES:
                        if feat not in opt_features:
                            fixed_values[feat] = base_formulation[feat]
                    
                    model = trained_models[target_choice]
                    
                    opt_indices = [FEATURES.index(feat) for feat in opt_features]
                    fixed_indices = [FEATURES.index(feat) for feat in fixed_values.keys()]
                    fixed_values_list = [fixed_values[feat] for feat in fixed_values.keys()]
                    
                    bounds = [(float(df[f].min()), float(df[f].max())) for f in opt_features]
                    median_values = df[FEATURES].median().values
                    
                    def objective_func_fast(x):
                        x_full = median_values.copy()
                        for idx, val in zip(opt_indices, x):
                            x_full[idx] = val
                        for idx, val in zip(fixed_indices, fixed_values_list):
                            x_full[idx] = val
                        
                        pred = model.predict([x_full])[0]
                        
                        if goal_type == "Maximize":
                            return -pred
                        elif goal_type == "Minimize":
                            return pred
                        else:
                            return abs(pred - target_val)
                    
                    res = differential_evolution(
                        objective_func_fast, 
                        bounds, 
                        seed=RANDOM_SEED, 
                        maxiter=40,
                        popsize=12,
                        tol=0.01,
                        mutation=(0.5, 1.0),
                        recombination=0.7,
                        workers=1,
                        disp=False
                    )
                    
                    if res.x is not None:
                        st.markdown("""
                        <div class="opt-result-card">
                            <div style="display:flex;align-items:center;gap:16px;margin-bottom:12px;">
                                <span style="font-size:2.2rem;">✨</span>
                                <div>
                                    <div style="font-weight:800;font-size:1.3rem;">Optimal Formulation Identified</div>
                                    <div style="font-size:0.85rem;font-weight:500;">Optimization completed successfully</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        opt_results_data = []
                        for i, feat in enumerate(opt_features):
                            category = EXCIPIENT_TO_CATEGORY.get(feat, "Other")
                            opt_results_data.append({
                                "Formulation Component": feat,
                                "Category": category,
                                "Optimized Concentration": round(res.x[i], 3),
                                "Lower Bound": round(bounds[i][0], 3),
                                "Upper Bound": round(bounds[i][1], 3)
                            })
                        
                        opt_res = pd.DataFrame(opt_results_data)
                        st.dataframe(opt_res, use_container_width=True, hide_index=True)
                        
                        final_formulation = base_formulation.copy()
                        for i, feat in enumerate(opt_features):
                            final_formulation[feat] = res.x[i]
                            
                        opt_df = pd.DataFrame([final_formulation])[FEATURES]
                        opt_df = opt_df.fillna(df[FEATURES].median())
                        opt_pred = float(model.predict(opt_df)[0])
                        
                        st.markdown(f"""
                        <div class="prediction-box">
                            <span>📊 Predicted {target_choice_display} for Optimized Formulation:</span>
                            <span>{opt_pred:.3f}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("---")
                        st.markdown("""
                        <div style="font-weight:700;font-size:1.1rem;color:var(--text-color-primary, #0f172a);margin-bottom:0.5rem;">
                            💾 Export Optimized Results
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col_opt_fn1, col_opt_fn2 = st.columns([2, 1])
                        with col_opt_fn1:
                            custom_filename_opt = st.text_input("File name:", value="the_formula_optimized_results", key="custom_filename_opt")
                        with col_opt_fn2:
                            opt_results_df = opt_res.copy()
                            opt_results_df["Predicted_Value"] = opt_pred
                            st.markdown(get_download_link(opt_results_df, custom_filename_opt), unsafe_allow_html=True)
                        
                    else:
                        st.error("❌ Optimization did not return valid parameters. Please try again.")
                        
                except Exception as e:
                    st.error(f"❌ Optimization failed: {str(e)}")
                    st.exception(e)


# ============================================================
# TAB 3: MODEL PERFORMANCE
# ============================================================

with tab3:
    st.markdown("""
    <div class="section-header">
        <span class="section-header-icon">📈</span>
        Model Performance Analytics
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="info-text">
        📊 Evaluation metrics for the machine learning models developed for each Critical Quality Attribute (CQA).
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
                
                display_name = TARGET_DISPLAY_NAMES.get(target, target.replace("_", " ").title())
                perf_records.append({
                    "Critical Quality Attribute": display_name,
                    "R² Score": round(r2, 3),
                    "RMSE": round(rmse, 3),
                    "MAE": round(mae, 3)
                })
            except Exception:
                continue
            
    if perf_records:
        perf_df = pd.DataFrame(perf_records)
        st.dataframe(perf_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.markdown("""
        <div style="font-weight:700;font-size:1.1rem;color:var(--text-color-primary, #0f172a);margin-bottom:1rem;">
            📊 Model Summary Statistics
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            avg_r2 = perf_df["R² Score"].mean()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{avg_r2:.3f}</div>
                <div class="metric-label">Average R² Score</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            avg_rmse = perf_df["RMSE"].mean()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{avg_rmse:.3f}</div>
                <div class="metric-label">Average RMSE</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            avg_mae = perf_df["MAE"].mean()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{avg_mae:.3f}</div>
                <div class="metric-label">Average MAE</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No performance metrics available.")
    
    # ============================================================
    # 📊 FORMULATION FACTORS ANALYSIS
    # ============================================================
    
    st.markdown("---")
    st.markdown("""
    <div class="section-header">
        <span class="section-header-icon">🔑</span>
        Critical Formulation Factors Analysis
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="info-text">
        📈 Relative importance of each formulation parameter in predicting the selected Critical Quality Attribute.
    </div>
    """, unsafe_allow_html=True)
    
    if feature_importance_data:
        available_targets = [t for t in TARGETS if t in feature_importance_data]
        display_options = [TARGET_DISPLAY_NAMES.get(t, t.replace("_", " ").title()) for t in available_targets]
        
        selected_target_display = st.selectbox(
            "Select Critical Quality Attribute (CQA)",
            options=display_options,
            key="importance_target_select",
            help="Select a quality attribute to visualize the relative importance of formulation factors."
        )
        
        reverse_display_map = {v: k for k, v in TARGET_DISPLAY_NAMES.items()}
        selected_target = reverse_display_map.get(selected_target_display, selected_target_display)
        
        if selected_target in feature_importance_data:
            importance_dict = feature_importance_data[selected_target]
            
            all_importance_df = pd.DataFrame(
                sorted(importance_dict.items(), key=lambda x: x[1], reverse=True),
                columns=["Formulation Factor", "Relative Importance (%)"]
            )
            
            fig = px.bar(
                all_importance_df.head(15),
                x="Relative Importance (%)",
                y="Formulation Factor",
                orientation='h',
                title=f"Formulation Factor Importance for {selected_target_display}",
                color="Relative Importance (%)",
                color_continuous_scale="Blues",
                text="Relative Importance (%)"
            )
            
            is_dark = st.get_option('theme.base') == 'dark'
            
            plot_bgcolor = 'rgba(0,0,0,0)'
            paper_bgcolor = 'rgba(0,0,0,0)'
            font_color = '#f1f5f9' if is_dark else '#0f172a'
            grid_color = 'rgba(255,255,255,0.1)' if is_dark else '#f1f5f9'
            text_color = '#94a3b8' if is_dark else '#475569'
            
            fig.update_traces(
                texttemplate='%{text:.1f}%', 
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Relative Importance: %{x:.1f}%<extra></extra>',
                marker=dict(
                    color=all_importance_df["Relative Importance (%)"],
                    colorscale="Blues",
                    showscale=False
                )
            )
            
            fig.update_layout(
                height=500,
                margin=dict(l=0, r=0, t=50, b=0),
                xaxis_title="Relative Importance (%)",
                yaxis_title=None,
                plot_bgcolor=plot_bgcolor,
                paper_bgcolor=paper_bgcolor,
                font=dict(family="Inter, sans-serif", size=12, color=font_color),
                title_font=dict(size=16, weight=700, color=font_color),
                xaxis=dict(
                    gridcolor=grid_color,
                    gridwidth=1,
                    tickfont=dict(size=11, color=text_color)
                ),
                yaxis=dict(
                    tickfont=dict(size=11, color=text_color)
                ),
                coloraxis_showscale=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("Feature importance data not available for the selected CQA.")
    else:
        st.info("Feature importance analysis is not available — this may be due to model compatibility limitations.")


# ============================================================
# 👨‍🔬 PROFESSIONAL FOOTER
# ============================================================

st.markdown(f"""
<div style="
    margin-top: 2.5rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--border-color, #e2e8f0);
    text-align: center;
    font-size: 0.78rem;
    color: var(--text-color-muted, #94a3b8);
    font-weight: 400;
    letter-spacing: 0.2px;
    line-height: 1.6;
">
    🧪 The Formula — AI Formulation Lab
    <span style="margin: 0 6px; opacity: 0.5;">|</span>
    Designed &amp; Developed by
    <strong style="color: var(--text-color-secondary, #475569); font-weight: 600;">Ziad Ibrahim</strong>
    <span style="margin: 0 6px; opacity: 0.5;">|</span>
    © {datetime.now().year} All Rights Reserved
</div>
""", unsafe_allow_html=True)
