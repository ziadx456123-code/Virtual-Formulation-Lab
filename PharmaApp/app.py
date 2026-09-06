from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from scipy.optimize import differential_evolution

from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import (
    train_test_split,
    RepeatedKFold,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Virtual Formulation Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed",
)

RANDOM_SEED = 42

TEST_SIZE = 0.20

CV_FOLDS = 5
CV_REPEATS = 2

# Reduced from 10 to 5 to speed up optimization
MAX_OPT_VARS = 5

OPT_LOW_PERCENTILE = 5
OPT_HIGH_PERCENTILE = 95

MODEL_VERSION = "3.0"

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "saved_cqa_models"
MODEL_DIR.mkdir(exist_ok=True)

META_FILE = MODEL_DIR / "training_metadata.json"

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
# PROFESSIONAL UI
# ============================================================

st.markdown(
    """
    <style>

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

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
    }

    .result-label {
        color: #64748b;
        font-size: 0.82rem;
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
# DATASET FINGERPRINT
# ============================================================

def get_dataset_path() -> Path:

    for path in DATASET_CANDIDATES:

        if path.exists():
            return path

    raise FileNotFoundError(
        "Required formulation dataset was not found."
    )


def calculate_file_hash(path: Path) -> str:

    sha256 = hashlib.sha256()

    with open(path, "rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):

            sha256.update(chunk)

    return sha256.hexdigest()


def get_training_signature(path: Path) -> str:

    signature_data = {

        "model_version": MODEL_VERSION,

        "dataset_hash": calculate_file_hash(path),

        "targets": TARGETS,

        "test_size": TEST_SIZE,

        "cv_folds": CV_FOLDS,

        "cv_repeats": CV_REPEATS,

        "random_seed": RANDOM_SEED,

    }

    raw = json.dumps(
        signature_data,
        sort_keys=True
    ).encode()

    return hashlib.sha256(
        raw
    ).hexdigest()


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:

    path = get_dataset_path()

    if path.suffix.lower() == ".csv":

        df = pd.read_csv(path)

    else:

        df = pd.read_excel(path)

    return clean_dataset(df)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    for col in df.columns:

        if df[col].dtype == "object":

            converted = pd.to_numeric(
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False),
                errors="coerce",
            )

            if converted.notna().mean() >= 0.80:

                df[col] = converted

    missing_targets = [
        c
        for c in TARGETS
        if c not in df.columns
    ]

    if missing_targets:

        raise ValueError(
            f"Required CQA columns are missing: "
            f"{missing_targets}"
        )

    numeric_cols = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    predictors = [
        c
        for c in numeric_cols
        if c not in TARGETS
    ]

    df = (
        df[
            predictors + TARGETS
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .drop_duplicates()
    )

    df = df.dropna(
        subset=TARGETS,
        how="all"
    )

    useful_predictors = []

    for col in predictors:

        if df[col].notna().sum() == 0:
            continue

        if df[col].nunique(
            dropna=True
        ) <= 1:
            continue

        useful_predictors.append(col)

    df = df[
        useful_predictors + TARGETS
    ]

    return df


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
) -> Dict[str, float]:

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    return {

        "R2": float(
            r2_score(
                y_true,
                y_pred
            )
        ),

        "MAE": float(
            mean_absolute_error(
                y_true,
                y_pred
            )
        ),

        "RMSE": float(rmse),
    }


# ============================================================
# MODEL CANDIDATES
# ============================================================

def get_model_candidates():

    return {

        "Extra Trees": (

            ExtraTreesRegressor(
                random_state=RANDOM_SEED,
                n_jobs=-1
            ),

            {

                "model__n_estimators": [
                    300,
                    500,
                    800
                ],

                "model__max_depth": [
                    None,
                    8,
                    12,
                    20
                ],

                "model__min_samples_split": [
                    2,
                    3,
                    5
                ],

                "model__min_samples_leaf": [
                    1,
                    2,
                    4
                ],

                "model__max_features": [
                    1.0,
                    "sqrt",
                    0.7
                ],

            }
        ),

        "Random Forest": (

            RandomForestRegressor(
                random_state=RANDOM_SEED,
                n_jobs=-1
            ),

            {

                "model__n_estimators": [
                    300,
                    500,
                    800
                ],

                "model__max_depth": [
                    None,
                    8,
                    12,
                    20
                ],

                "model__min_samples_split": [
                    2,
                    4,
                    6
                ],

                "model__min_samples_leaf": [
                    1,
                    2,
                    4
                ],

                "model__max_features": [
                    1.0,
                    "sqrt",
                    0.7
                ],

            }
        ),

        "Gradient Boosting": (

            GradientBoostingRegressor(
                random_state=RANDOM_SEED
            ),

            {

                "model__n_estimators": [
                    100,
                    200,
                    300
                ],

                "model__learning_rate": [
                    0.02,
                    0.05,
                    0.08,
                    0.1
                ],

                "model__max_depth": [
                    2,
                    3,
                    4
                ],

                "model__min_samples_leaf": [
                    1,
                    2,
                    4
                ],

                "model__subsample": [
                    0.8,
                    1.0
                ],

            }
        ),

        "HistGradient Boosting": (

            HistGradientBoostingRegressor(
                random_state=RANDOM_SEED
            ),

            {

                "model__max_iter": [
                    100,
                    200,
                    300
                ],

                "model__learning_rate": [
                    0.03,
                    0.05,
                    0.1
                ],

                "model__max_leaf_nodes": [
                    15,
                    31,
                    63
                ],

                "model__l2_regularization": [
                    0.0,
                    0.1,
                    1.0
                ],

                "model__min_samples_leaf": [
                    10,
                    15,
                    20
                ],

            }
        ),

        "Ridge": (

            Ridge(),

            {

                "model__alpha": [
                    0.01,
                    0.1,
                    1,
                    10,
                    100
                ]

            }
        ),
    }


# ============================================================
# PIPELINE
# ============================================================

def make_pipeline(model):

    return Pipeline(

        steps=[

            (
                "imputer",

                SimpleImputer(
                    strategy="median",
                    add_indicator=True
                )
            ),

            (
                "scaler",

                StandardScaler()
            ),

            (
                "model",

                model
            ),

        ]
    )


# ============================================================
# MODEL FILES
# ============================================================

def model_file_for(target: str) -> Path:

    safe_name = (
        target
        .replace(" ", "_")
        .replace("/", "_")
    )

    return (
        MODEL_DIR
        / f"{safe_name}_bundle.joblib"
    )


def models_are_valid(
    signature: str
) -> bool:

    if not META_FILE.exists():
        return False

    try:

        with open(
            META_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            metadata = json.load(f)

        if (
            metadata.get(
                "training_signature"
            )
            != signature
        ):

            return False

        for target in TARGETS:

            if not model_file_for(
                target
            ).exists():

                return False

        return True

    except Exception:

        return False


def load_saved_models():

    trained_models = {}

    for target in TARGETS:

        trained_models[target] = (
            joblib.load(
                model_file_for(
                    target
                )
            )
        )

    return trained_models


# ============================================================
# TRAINING
# ============================================================

@st.cache_resource(
    show_spinner=True
)
def train_all_models(
    training_signature: str
):

    # --------------------------------------------------------
    # LOAD EXISTING MODELS
    # --------------------------------------------------------

    if models_are_valid(
        training_signature
    ):

        try:

            return load_saved_models()

        except Exception:

            pass

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    df = load_dataset()

    trained_models = {}

    for target in TARGETS:

        X = df.drop(
            columns=TARGETS
        )

        y = df[target]

        valid = y.notna()

        X = X.loc[
            valid
        ].copy()

        y = y.loc[
            valid
        ].copy()

        if len(y) < 15:

            raise ValueError(
                f"Not enough experimental "
                f"observations for {target}."
            )

        # ----------------------------------------------------
        # TEST SET
        # ----------------------------------------------------

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=TEST_SIZE,
                random_state=RANDOM_SEED
            )
        )

        # ----------------------------------------------------
        # ADAPT CV TO DATA SIZE
        # ----------------------------------------------------

        n_splits = min(
            CV_FOLDS,
            len(y_train)
        )

        if n_splits < 2:

            raise ValueError(
                f"Not enough training observations "
                f"for validation of {target}."
            )

        cv = RepeatedKFold(

            n_splits=n_splits,

            n_repeats=CV_REPEATS,

            random_state=RANDOM_SEED
        )

        candidates = (
            get_model_candidates()
        )

        best_pipeline = None

        best_cv_score = -np.inf

        best_model_name = None

        # ----------------------------------------------------
        # MODEL SELECTION
        # ----------------------------------------------------

        for model_name, (
            model,
            params
        ) in candidates.items():

            pipeline = (
                make_pipeline(
                    model
                )
            )

            total_combinations = int(
                np.prod(
                    [
                        len(v)
                        for v in params.values()
                    ]
                )
            )

            n_iter = min(
                15,
                total_combinations
            )

            search = (
                RandomizedSearchCV(

                    estimator=pipeline,

                    param_distributions=params,

                    n_iter=n_iter,

                    scoring="r2",

                    cv=cv,

                    random_state=RANDOM_SEED,

                    n_jobs=-1,

                    error_score=np.nan,

                    refit=True
                )
            )

            try:

                search.fit(
                    X_train,
                    y_train
                )

                score = (
                    search.best_score_
                )

                if (
                    np.isfinite(score)
                    and score > best_cv_score
                ):

                    best_cv_score = score

                    best_pipeline = (
                        search.best_estimator_
                    )

                    best_model_name = (
                        model_name
                    )

            except Exception:

                continue

        # ----------------------------------------------------
        # FALLBACK
        # ----------------------------------------------------

        if best_pipeline is None:

            best_pipeline = (
                make_pipeline(

                    ExtraTreesRegressor(

                        n_estimators=500,

                        random_state=RANDOM_SEED,

                        n_jobs=-1
                    )
                )
            )

            best_pipeline.fit(
                X_train,
                y_train
            )

            best_model_name = (
                "Extra Trees"
            )

            best_cv_score = np.nan

        # ----------------------------------------------------
        # TEST PREDICTION
        # ----------------------------------------------------

        y_pred = (
            best_pipeline.predict(
                X_test
            )
        )

        metrics = calculate_metrics(
            y_test,
            y_pred
        )

        metrics["CV R2"] = (
            float(best_cv_score)
            if np.isfinite(
                best_cv_score
            )
            else np.nan
        )

        # ----------------------------------------------------
        # FEATURE IMPORTANCE
        # ----------------------------------------------------

        try:

            perm = (
                permutation_importance(

                    best_pipeline,

                    X_train,

                    y_train,

                    scoring="r2",

                    n_repeats=10,

                    random_state=RANDOM_SEED,

                    n_jobs=-1
                )
            )

            importance = pd.DataFrame(

                {

                    "Feature":
                        X_train.columns,

                    "Importance":
                        np.maximum(
                            perm.importances_mean,
                            0
                        ),

                }
            )

            total = (
                importance[
                    "Importance"
                ].sum()
            )

            if total > 0:

                importance[
                    "Importance_%"
                ] = (

                    importance[
                        "Importance"
                    ]

                    / total

                    * 100

                )

            else:

                importance[
                    "Importance_%"
                ] = 0.0

            importance = (
                importance
                .sort_values(
                    "Importance_%",
                    ascending=False
                )
                .reset_index(
                    drop=True
                )
            )

        except Exception:

            importance = pd.DataFrame(

                columns=[

                    "Feature",

                    "Importance",

                    "Importance_%"

                ]
            )

        # ----------------------------------------------------
        # STORE
        # ----------------------------------------------------

        trained_models[target] = {

            "model":
                best_pipeline,

            "target":
                target,

            "features":
                X.columns.tolist(),

            "metrics":
                metrics,

            "importance":
                importance,

            "model_name":
                best_model_name,

            "X_test":
                X_test,

            "y_test":
                y_test,

            "y_pred":
                y_pred,
        }

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        joblib.dump(

            trained_models[target],

            model_file_for(
                target
            ),

            compress=3
        )

    # --------------------------------------------------------
    # SAVE METADATA
    # --------------------------------------------------------

    metadata = {

        "training_signature":
            training_signature,

        "model_version":
            MODEL_VERSION,

        "targets":
            TARGETS,

    }

    with open(
        META_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

    return trained_models


# ============================================================
# PREDICTION
# ============================================================

def predict_cqas(
    models,
    inputs
):

    predictions = {}

    for target in TARGETS:

        value = (
            models[target][
                "model"
            ]
            .predict(inputs)[0]
        )

        predictions[target] = (
            float(value)
        )

    return predictions


# ============================================================
# INPUT RANGES
# ============================================================

def get_input_ranges(df):

    ranges = {}

    for col in df.columns:

        if col in TARGETS:
            continue

        series = (
            pd.to_numeric(
                df[col],
                errors="coerce"
            )
            .dropna()
        )

        if len(series) == 0:
            continue

        ranges[col] = {

            "min":
                float(series.min()),

            "max":
                float(series.max()),

            "p05":
                float(
                    series.quantile(
                        OPT_LOW_PERCENTILE
                        / 100
                    )
                ),

            "p95":
                float(
                    series.quantile(
                        OPT_HIGH_PERCENTILE
                        / 100
                    )
                ),

            "unique":
                int(
                    series.nunique()
                ),

            "values":
                sorted(
                    series.unique()
                    .tolist()
                ),
        }

    return ranges


# ============================================================
# FEATURE CLASSIFICATION
# ============================================================

def classify_features(
    input_columns
):

    physicochemical_keywords = [

        "Weight",
        "Density",
        "Index",
        "Ratio",
        "Repose",
        "Thickness",
        "Molecular",
        "XLogP3",
        "Hydrogen",
        "Rotational",
        "Topological",
        "Heavy",
        "Complexity",
        "LogS",
        "DOSE",
        "Wetting",
        "Flow",
        "Compressibility",
        "Porosity",
        "Moisture",
        "Particle",
        "Size",
    ]

    phys_cols = []

    excipient_cols = []

    for col in input_columns:

        if any(

            kw.lower()
            in col.lower()

            for kw in
            physicochemical_keywords

        ):

            phys_cols.append(
                col
            )

        else:

            excipient_cols.append(
                col
            )

    return (
        phys_cols,
        excipient_cols
    )


# ============================================================
# DESIRABILITY
# ============================================================

def desirability(
    value,
    goal,
    low,
    high,
    target=None
):

    if not np.isfinite(value):
        return 0.0

    if high <= low:
        return 0.0

    if goal == "Minimize":

        if value <= low:
            return 1.0

        if value >= high:
            return 0.0

        return (
            high - value
        ) / (
            high - low
        )

    if goal == "Maximize":

        if value >= high:
            return 1.0

        if value <= low:
            return 0.0

        return (
            value - low
        ) / (
            high - low
        )

    target = float(

        np.clip(

            target
            if target is not None
            else (low + high) / 2,

            low,

            high

        )
    )

    if value == target:
        return 1.0

    if value < target:

        if target == low:
            return 0.0

        return max(

            0.0,

            (
                value - low
            )
            /
            (
                target - low
            )

        )

    if high == target:
        return 0.0

    return max(

        0.0,

        (
            high - value
        )
        /
        (
            high - target
        )

    )


def overall_desirability(
    predictions,
    goals,
    df
):

    values = []

    for target in TARGETS:

        series = (
            df[target]
            .dropna()
        )

        if len(series) == 0:
            continue

        score = desirability(

            predictions[target],

            goals[target]["goal"],

            float(series.min()),

            float(series.max()),

            goals[target].get(
                "target"
            )

        )

        values.append(
            np.clip(
                score,
                0,
                1
            )
        )

    if not values:
        return 0.0

    return float(

        np.prod(values)
        **
        (
            1 / len(values)
        )

    )


# ============================================================
# OPTIMIZATION VARIABLE SELECTION
# ============================================================

def select_optimization_variables(
    models
):

    score_map = {}

    for target in TARGETS:

        importance = (
            models[target][
                "importance"
            ]
        )

        for _, row in (
            importance.iterrows()
        ):

            feature = row[
                "Feature"
            ]

            score_map[feature] = (

                score_map.get(
                    feature,
                    0
                )

                +

                float(
                    row[
                        "Importance_%"
                    ]
                )

            )

    ranked = sorted(

        score_map.items(),

        key=lambda x: x[1],

        reverse=True

    )

    return [

        feature

        for feature, _ in
        ranked[
            :MAX_OPT_VARS
        ]

    ]


# ============================================================
# OPTIMIZATION
# ============================================================

def optimize_formulation(
    df,
    models,
    baseline_inputs,
    goals
):

    ranges = (
        get_input_ranges(
            df
        )
    )

    selected = [

        x

        for x in
        select_optimization_variables(
            models
        )

        if x in ranges

    ]

    if not selected:

        return (

            baseline_inputs.copy(),

            {},

            0.0

        )

    base = (
        baseline_inputs.copy()
    )

    bounds = []

    for feature in selected:

        info = ranges[
            feature
        ]

        low = info["p05"]

        high = info["p95"]

        if low >= high:

            low = info["min"]

            high = info["max"]

        bounds.append(
            (
                low,
                high
            )
        )

    discrete = {

        feature:
            ranges[feature][
                "values"
            ]

        for feature in selected

        if ranges[feature][
            "unique"
        ] <= 10

    }

    reference_X = (
        df[
            [
                f
                for f in selected
                if f in df.columns
            ]
        ]
        .dropna()
    )

    def objective(vals):

        candidate = (
            base.copy()
        )

        for feature, value in zip(
            selected,
            vals
        ):

            if feature in discrete:

                arr = np.asarray(

                    discrete[
                        feature
                    ],

                    dtype=float

                )

                value = float(

                    arr[
                        np.argmin(
                            np.abs(
                                arr - value
                            )
                        )
                    ]

                )

            candidate[
                feature
            ] = value

        predictions = (
            predict_cqas(
                models,
                candidate
            )
        )

        desirability_score = (
            overall_desirability(

                predictions,

                goals,

                df

            )
        )

        penalty = 0.0

        if len(reference_X) > 0:

            candidate_values = (

                candidate[
                    selected
                ]
                .astype(float)
                .iloc[0]
                .values

            )

            ref_values = (

                reference_X
                .astype(float)
                .values

            )

            scale = (

                np.nanstd(
                    ref_values,
                    axis=0
                )

                + 1e-9

            )

            distances = np.sqrt(

                np.nanmean(

                    (

                        (
                            ref_values
                            -
                            candidate_values
                        )

                        /

                        scale

                    )

                    ** 2,

                    axis=1

                )

            )

            nearest_distance = float(

                np.min(
                    distances
                )
            )

            penalty = (

                0.03

                *

                max(

                    0.0,

                    nearest_distance
                    - 2.0

                )

            )

        return -(

            desirability_score
            - penalty

        )

    # --------------------------------------------------------
    # FASTER DIFFERENTIAL EVOLUTION
    # --------------------------------------------------------

    result = (
        differential_evolution(

            objective,

            bounds=bounds,

            seed=RANDOM_SEED,

            # Reduced from 80
            maxiter=30,

            # Reduced from 10
            popsize=6,

            # Relaxed from 1e-7
            tol=1e-4,

            mutation=(0.5, 1.0),

            recombination=0.7,

            polish=True,

            workers=1

        )
    )

    opt_inputs = (
        base.copy()
    )

    for feature, value in zip(
        selected,
        result.x
    ):

        if feature in discrete:

            arr = np.asarray(

                discrete[
                    feature
                ],

                dtype=float

            )

            value = float(

                arr[
                    np.argmin(
                        np.abs(
                            arr - value
                        )
                    )
                ]

            )

        opt_inputs[
            feature
        ] = value

    opt_predictions = (
        predict_cqas(
            models,
            opt_inputs
        )
    )

    final_desirability = (
        overall_desirability(

            opt_predictions,

            goals,

            df

        )
    )

    return (

        opt_inputs,

        opt_predictions,

        final_desirability

    )


# ============================================================
# APPLICATION INITIALIZATION
# ============================================================

try:

    dataset_path = (
        get_dataset_path()
    )

    training_signature = (
        get_training_signature(
            dataset_path
        )
    )

    df = load_dataset()

    models = train_all_models(
        training_signature
    )

except Exception as e:

    st.error(
        "⚠️ The application could not initialize. "
        f"Error: {e}"
    )

    st.stop()


input_columns = [

    c

    for c in df.columns

    if c not in TARGETS

]

input_ranges = (
    get_input_ranges(
        df
    )
)

phys_cols, excipient_cols = (
    classify_features(
        input_columns
    )
)


# ============================================================
# HEADER
# ============================================================

st.markdown(

    '<div class="main-title">'
    '🧪 Virtual Formulation Lab'
    '</div>',

    unsafe_allow_html=True

)

st.markdown(

    '<div class="subtitle">'
    'AI-Assisted Pharmaceutical Formulation '
    '& Quality Prediction Platform'
    '</div>',

    unsafe_allow_html=True

)


# ============================================================
# MAIN TABS
# ============================================================

tab_lab, tab_optimizer, tab_validation = (
    st.tabs(
        [
            "🧪 Virtual Lab",
            "🎯 Optimization",
            "✅ Validation",
        ]
    )
)


# ============================================================
# TAB 1 — VIRTUAL LAB
# ============================================================

with tab_lab:

    st.markdown(

        '<div class="section-title">'
        'Formulation Setup'
        '</div>',

        unsafe_allow_html=True

    )

    formulation_name = st.text_input(

        "Formulation Batch Code",

        value="Batch_01"

    )

    st.markdown(

        '<div class="info-box">'
        'Enter the formulation and powder properties. '
        'The system will predict the expected Critical '
        'Quality Attributes (CQAs).'
        '</div>',

        unsafe_allow_html=True

    )

    values = {}

    # --------------------------------------------------------
    # PHYSICOCHEMICAL
    # --------------------------------------------------------

    with st.expander(

        "⚙️ Physicochemical & Powder Properties",

        expanded=True

    ):

        if phys_cols:

            for start in range(

                0,

                len(phys_cols),

                3

            ):

                cols = st.columns(3)

                for col_idx, feature in enumerate(

                    phys_cols[
                        start:start + 3
                    ]

                ):

                    info = input_ranges[
                        feature
                    ]

                    min_v = info[
                        "min"
                    ]

                    max_v = info[
                        "max"
                    ]

                    with cols[
                        col_idx
                    ]:

                        if min_v == max_v:

                            values[
                                feature
                            ] = min_v

                            st.number_input(

                                f"{feature} (Fixed)",

                                value=float(
                                    min_v
                                ),

                                disabled=True,

                                key=f"inp_{feature}"

                            )

                        elif info[
                            "unique"
                        ] <= 10:

                            values[
                                feature
                            ] = (

                                st.selectbox(

                                    feature,

                                    options=info[
                                        "values"
                                    ],

                                    key=f"inp_{feature}"

                                )

                            )

                        else:

                            values[
                                feature
                            ] = (

                                st.number_input(

                                    feature,

                                    min_value=float(
                                        min_v
                                    ),

                                    max_value=float(
                                        max_v
                                    ),

                                    value=float(

                                        (
                                            min_v
                                            +
                                            max_v
                                        )
                                        / 2

                                    ),

                                    format="%.4f",

                                    key=f"inp_{feature}"

                                )

                            )

    # --------------------------------------------------------
    # EXCIPIENTS
    # --------------------------------------------------------

    with st.expander(

        "💊 Excipients Composition",

        expanded=False

    ):

        if excipient_cols:

            for start in range(

                0,

                len(excipient_cols),

                3

            ):

                cols = st.columns(3)

                for col_idx, feature in enumerate(

                    excipient_cols[
                        start:start + 3
                    ]

                ):

                    info = input_ranges[
                        feature
                    ]

                    min_v = info[
                        "min"
                    ]

                    max_v = info[
                        "max"
                    ]

                    with cols[
                        col_idx
                    ]:

                        if min_v == max_v:

                            values[
                                feature
                            ] = min_v

                            st.number_input(

                                f"{feature} (Fixed)",

                                value=float(
                                    min_v
                                ),

                                disabled=True,

                                key=f"exc_{feature}"

                            )

                        elif info[
                            "unique"
                        ] <= 10:

                            values[
                                feature
                            ] = (

                                st.selectbox(

                                    feature,

                                    options=info[
                                        "values"
                                    ],

                                    key=f"exc_{feature}"

                                )

                            )

                        else:

                            values[
                                feature
                            ] = (

                                st.number_input(

                                    feature,

                                    min_value=float(
                                        min_v
                                    ),

                                    max_value=float(
                                        max_v
                                    ),

                                    value=float(
                                        min_v
                                    ),

                                    format="%.4f",

                                    key=f"exc_{feature}"

                                )

                            )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if st.button(

        "🔬 Predict CQAs",

        type="primary",

        use_container_width=True

    ):

        baseline_inputs = pd.DataFrame(

            [values],

            columns=input_columns

        )

        predictions = (
            predict_cqas(

                models,

                baseline_inputs

            )
        )

        st.session_state[
            "baseline_inputs"
        ] = baseline_inputs

        st.session_state[
            "predictions"
        ] = predictions

        st.session_state.pop(
            "opt_preds",
            None
        )

        st.session_state.pop(
            "opt_inputs",
            None
        )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    if (
        "predictions"
        in st.session_state
    ):

        st.markdown(

            '<div class="section-title">'
            'Predicted CQAs'
            '</div>',

            unsafe_allow_html=True

        )

        res_cols = st.columns(
            len(TARGETS)
        )

        for i, target in enumerate(
            TARGETS
        ):

            value = (
                st.session_state[
                    "predictions"
                ][target]
            )

            with res_cols[i]:

                st.markdown(

                    f"""
                    <div class="result-card">
                        <div class="result-label">
                            {target}
                        </div>
                        <div class="result-value">
                            {value:.3f}
                        </div>
                    </div>
                    """,

                    unsafe_allow_html=True

                )

        st.markdown(
            "<br>",
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # DOWNLOAD PREDICTIONS
        # ----------------------------------------------------

        pred_export_df = pd.DataFrame(
            [
                st.session_state[
                    "predictions"
                ]
            ]
        )

        pred_export_df.insert(
            0,
            "Batch_Code",
            formulation_name
        )

        csv_data = (
            pred_export_df
            .to_csv(
                index=False
            )
            .encode("utf-8")
        )

        st.download_button(

            label="📥 Download Predicted CQAs (CSV)",

            data=csv_data,

            file_name=(
                f"{formulation_name}"
                f"_predictions.csv"
            ),

            mime="text/csv",

            use_container_width=True

        )


# ============================================================
# TAB 2 — OPTIMIZATION
# ============================================================

with tab_optimizer:

    st.markdown(

        '<div class="section-title">'
        'CQA Optimization Targets'
        '</div>',

        unsafe_allow_html=True

    )

    goals = {}

    goal_cols = st.columns(2)

    for i, target in enumerate(
        TARGETS
    ):

        with goal_cols[
            i % 2
        ]:

            goal = st.selectbox(

                f"{target}",

                [
                    "Target",
                    "Minimize",
                    "Maximize"
                ],

                key=f"goal_{target}"

            )

            target_value = None

            if goal == "Target":

                series = (
                    df[target]
                    .dropna()
                )

                target_value = (
                    st.number_input(

                        f"Desired {target}",

                        min_value=float(
                            series.min()
                        ),

                        max_value=float(
                            series.max()
                        ),

                        value=float(
                            series.median()
                        ),

                        key=f"target_{target}"

                    )
                )

            goals[target] = {

                "goal":
                    goal,

                "target":
                    target_value

            }

    if st.button(

        "🚀 Optimize Formulation",

        type="primary",

        use_container_width=True

    ):

        if (
            "baseline_inputs"
            not in st.session_state
        ):

            st.warning(

                "Please predict the current "
                "formulation first."

            )

        else:

            with st.spinner(

                "Searching for the best formulation..."

            ):

                (

                    opt_inputs,

                    opt_predictions,

                    desirability_score

                ) = optimize_formulation(

                    df,

                    models,

                    st.session_state[
                        "baseline_inputs"
                    ],

                    goals

                )

            st.session_state[
                "opt_inputs"
            ] = opt_inputs

            st.session_state[
                "opt_preds"
            ] = opt_predictions

            st.session_state[
                "opt_des"
            ] = desirability_score

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    if (
        "opt_preds"
        in st.session_state
    ):

        st.markdown(

            '<div class="section-title">'
            'Recommended Formulation'
            '</div>',

            unsafe_allow_html=True

        )

        opt_inputs = (
            st.session_state[
                "opt_inputs"
            ]
        )

        formulation_table = pd.DataFrame(

            {

                "Component":
                    input_columns,

                "Recommended Value": [

                    float(

                        opt_inputs.iloc[
                            0
                        ][col]

                    )

                    for col
                    in input_columns

                ],

            }
        )

        st.dataframe(

            formulation_table,

            use_container_width=True,

            hide_index=True

        )

        st.markdown(

            '<div class="section-title">'
            'Predicted CQAs After Optimization'
            '</div>',

            unsafe_allow_html=True

        )

        comparison = pd.DataFrame(

            {

                "CQA":
                    TARGETS,

                "Current Prediction": [

                    st.session_state[
                        "predictions"
                    ][t]

                    for t in TARGETS

                ],

                "Optimized Prediction": [

                    st.session_state[
                        "opt_preds"
                    ][t]

                    for t in TARGETS

                ],

            }
        )

        st.dataframe(

            comparison,

            use_container_width=True,

            hide_index=True

        )

        st.metric(

            "Overall Desirability",

            f"{st.session_state['opt_des']:.3f}"

        )

        st.markdown(
            "<br>",
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # DOWNLOAD OPTIMIZED FORMULATION
        # ----------------------------------------------------

        csv_opt_data = (
            formulation_table
            .to_csv(
                index=False
            )
            .encode("utf-8")
        )

        st.download_button(

            label=(
                "📥 Download Recommended "
                "Formulation (CSV)"
            ),

            data=csv_opt_data,

            file_name=(
                f"{formulation_name}"
                f"_optimized_formulation.csv"
            ),

            mime="text/csv",

            use_container_width=True

        )


# ============================================================
# TAB 3 — VALIDATION
# ============================================================

with tab_validation:

    st.markdown(

        '<div class="section-title">'
        'Model Validation'
        '</div>',

        unsafe_allow_html=True

    )

    st.markdown(

        '<div class="info-box">'
        'Validation results are calculated on experimental '
        'observations kept separate from model training.'
        '</div>',

        unsafe_allow_html=True

    )

    validation_data = []

    for target in TARGETS:

        metrics = (
            models[target][
                "metrics"
            ]
        )

        validation_data.append(

            {

                "CQA":
                    target,

                "Test R²":
                    metrics["R2"],

                "Test RMSE":
                    metrics["RMSE"],

                "Test MAE":
                    metrics["MAE"],

            }
        )

    validation_df = pd.DataFrame(
        validation_data
    )

    st.dataframe(

        validation_df,

        use_container_width=True,

        hide_index=True

    )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.markdown(

        '<div class="section-title">'
        'Key Formulation Factors'
        '</div>',

        unsafe_allow_html=True

    )

    selected_cqa = st.selectbox(

        "Select CQA",

        TARGETS

    )

    importance_df = (

        models[selected_cqa][
            "importance"
        ]

        .head(10)

        .sort_values(

            "Importance_%",

            ascending=True

        )

    )

    if len(
        importance_df
    ) > 0:

        fig = px.bar(

            importance_df,

            x="Importance_%",

            y="Feature",

            orientation="h",

            title=(

                f"Key Factors Affecting "
                f"{selected_cqa}"

            ),

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )