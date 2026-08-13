"""Shared ML pipeline for Phoenix AI — used by app.py and the notebook."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from category_encoders import OneHotEncoder as CatOHE

    _OHE_ERROR: Optional[str] = None
except Exception as _exc:  # pragma: no cover - import guard
    CatOHE = None
    _OHE_ERROR = str(_exc)

# Columns that carry no usable signal: a unique identifier plus three constants.
UNINFORMATIVE_COLS = [
    "EmployeeNumber",
    "EmployeeCount",
    "StandardHours",
    "Over18",
]

# Columns that overlap heavily with tenure and role features we keep.
REDUNDANT_COLS = [
    "JobLevel",
    "PerformanceRating",
    "YearsWithCurrManager",
    "YearsInCurrentRole",
    "TotalWorkingYears",
]

DROP_COLS = UNINFORMATIVE_COLS + REDUNDANT_COLS

CAT_FEATURES = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime",
]

# Features whose absence makes batch predictions unreliable.
CRITICAL_FEATURES = [
    "Department",
    "JobRole",
    "OverTime",
    "MonthlyIncome",
    "Age",
    "YearsAtCompany",
]

RANDOM_STATE = 42
ENSEMBLE_THRESHOLD = 0.50


def score_model(
    y_true: pd.Series,
    pred: np.ndarray,
    proba: np.ndarray,
    leave_class: int,
) -> Dict[str, float]:
    """Accuracy / Precision / Recall / ROC-AUC for one model on one split."""
    y_bin = (y_true == leave_class).astype(int)
    return {
        "acc": float(accuracy_score(y_true, pred)),
        "precision": float(
            precision_score(y_true, pred, pos_label=leave_class, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, pred, pos_label=leave_class, zero_division=0)
        ),
        "roc": float(roc_auc_score(y_bin, proba)),
    }


def _require_ohe() -> None:
    if CatOHE is None:
        raise ImportError(
            "category_encoders is required. Run: pip install category_encoders\n"
            f"{_OHE_ERROR}"
        )


def train_models(df: pd.DataFrame) -> Dict[str, Any]:
    """Train LR + RF, evaluate on validation/test, return metrics and artifacts."""
    _require_ohe()

    work = df.copy()
    work["Attrition"] = work["Attrition"].map({"Yes": 1, "No": 0})
    work = work.drop(columns=DROP_COLS)
    X = work.drop("Attrition", axis=1)
    y = work["Attrition"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )
    y_train = y_train.astype(int)
    y_val = y_val.astype(int)
    y_test = y_test.astype(int)

    lr_pipe = make_pipeline(
        CatOHE(),
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    )
    lr_pipe.fit(X_train, y_train)

    rf_pipe = make_pipeline(
        CatOHE(),
        RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE),
    )
    rf_pipe.fit(X_train, y_train)

    leave_class = int(max(lr_pipe.named_steps["logisticregression"].classes_))
    stay_class = int(min(lr_pipe.named_steps["logisticregression"].classes_))

    lr_classes = list(lr_pipe.named_steps["logisticregression"].classes_)
    rf_classes = list(rf_pipe.named_steps["randomforestclassifier"].classes_)
    lr_leave_idx = lr_classes.index(leave_class)
    rf_leave_idx = rf_classes.index(leave_class)

    lr_val_pred = lr_pipe.predict(X_val)
    lr_val_proba = lr_pipe.predict_proba(X_val)[:, lr_leave_idx]
    rf_val_pred = rf_pipe.predict(X_val)
    rf_val_proba = rf_pipe.predict_proba(X_val)[:, rf_leave_idx]
    ens_val_proba = (lr_val_proba + rf_val_proba) / 2
    ens_val_pred = (ens_val_proba >= ENSEMBLE_THRESHOLD).astype(int)

    lr_val = score_model(y_val, lr_val_pred, lr_val_proba, leave_class)
    rf_val = score_model(y_val, rf_val_pred, rf_val_proba, leave_class)
    ens_val = score_model(y_val, ens_val_pred, ens_val_proba, leave_class)

    lr_pred = lr_pipe.predict(X_test)
    lr_proba = lr_pipe.predict_proba(X_test)[:, lr_leave_idx]
    rf_pred = rf_pipe.predict(X_test)
    rf_proba = rf_pipe.predict_proba(X_test)[:, rf_leave_idx]
    ens_proba = (lr_proba + rf_proba) / 2
    ens_pred = (ens_proba >= ENSEMBLE_THRESHOLD).astype(int)

    lr_test = score_model(y_test, lr_pred, lr_proba, leave_class)
    rf_test = score_model(y_test, rf_pred, rf_proba, leave_class)
    ens_test = score_model(y_test, ens_pred, ens_proba, leave_class)

    feat_imp = _feature_importance(rf_pipe)

    return {
        "lr_pipe": lr_pipe,
        "rf_pipe": rf_pipe,
        "feat_cols": list(X.columns),
        "lr_acc": lr_test["acc"],
        "rf_acc": rf_test["acc"],
        "ens_acc": ens_test["acc"],
        "lr_precision": lr_test["precision"],
        "rf_precision": rf_test["precision"],
        "ens_precision": ens_test["precision"],
        "lr_recall": lr_test["recall"],
        "rf_recall": rf_test["recall"],
        "ens_recall": ens_test["recall"],
        "lr_roc": lr_test["roc"],
        "rf_roc": rf_test["roc"],
        "ens_roc": ens_test["roc"],
        "lr_val": lr_val,
        "rf_val": rf_val,
        "ens_val": ens_val,
        "lr_pred": lr_pred,
        "rf_pred": rf_pred,
        "ens_pred": ens_pred,
        "lr_proba": lr_proba,
        "rf_proba": rf_proba,
        "ens_proba": ens_proba,
        "y_test": y_test.values,
        "y_test_bin": (y_test == leave_class).astype(int).values,
        "y_val": y_val.values,
        "leave_class": leave_class,
        "stay_class": stay_class,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "feat_imp": feat_imp,
        # Notebook compatibility — splits for cells that still reference them.
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
    }


def _feature_importance(rf_pipe) -> Optional[pd.DataFrame]:
    try:
        ohe = rf_pipe.named_steps["onehotencoder"]
        rf = rf_pipe.named_steps["randomforestclassifier"]
        try:
            enc_cols = list(ohe.get_feature_names_out())
        except AttributeError:
            enc_cols = list(ohe.feature_names)
        fi = pd.DataFrame({"feature": enc_cols, "importance": rf.feature_importances_})
        rows: List[Dict[str, Any]] = []
        for cat in CAT_FEATURES:
            mask = fi["feature"].str.startswith(cat)
            if mask.any():
                rows.append(
                    {"feature": cat, "importance": fi.loc[mask, "importance"].sum()}
                )
        num_mask = ~fi["feature"].apply(
            lambda f: any(f.startswith(c) for c in CAT_FEATURES)
        )
        rows.extend(fi[num_mask].to_dict("records"))
        return pd.DataFrame(rows).sort_values("importance", ascending=False).head(15)
    except Exception:
        return None


def resolve_value(raw: Dict[str, Any], col: str) -> Any:
    if col in raw:
        return raw[col]
    normalized = {str(k).strip().lower(): v for k, v in raw.items()}
    return normalized.get(str(col).strip().lower())


def build_row(raw: Dict[str, Any], feat_cols: List[str]) -> Dict[str, Any]:
    """Build one feature row with defaults for missing values."""
    row: Dict[str, Any] = {}
    for col in feat_cols:
        value = resolve_value(raw, col)
        if col in CAT_FEATURES:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                row[col] = "Unknown"
            else:
                row[col] = str(value)
        elif value is None or (isinstance(value, float) and pd.isna(value)):
            row[col] = 0.0
        else:
            try:
                row[col] = float(value)
            except (TypeError, ValueError):
                row[col] = 0.0
    return row


def build_input(raw: Dict[str, Any], mi: Dict[str, Any]) -> pd.DataFrame:
    feat_cols: List[str] = mi["feat_cols"]
    return pd.DataFrame([build_row(raw, feat_cols)], columns=feat_cols)


def assess_batch_columns(
    batch_df: pd.DataFrame, feat_cols: List[str]
) -> Dict[str, Any]:
    """Check uploaded CSV columns; flag missing required fields."""
    present = [c for c in feat_cols if c in batch_df.columns]
    missing = [c for c in feat_cols if c not in batch_df.columns]
    missing_critical = [c for c in CRITICAL_FEATURES if c in missing]
    return {
        "present": present,
        "missing": missing,
        "missing_critical": missing_critical,
        "can_predict": len(missing) == 0,
    }


def predict(mi: Dict[str, Any], raw: Dict[str, Any]) -> Dict[str, Any]:
    inp = build_input(raw, mi)
    lc = mi["leave_class"]

    lr_cls = mi["lr_pipe"].predict(inp)[0]
    lr_prob = mi["lr_pipe"].predict_proba(inp)[0]
    lr_cls_list = list(mi["lr_pipe"].named_steps["logisticregression"].classes_)
    lr_leave_p = float(lr_prob[lr_cls_list.index(lc)])

    rf_cls = mi["rf_pipe"].predict(inp)[0]
    rf_prob = mi["rf_pipe"].predict_proba(inp)[0]
    rf_cls_list = list(mi["rf_pipe"].named_steps["randomforestclassifier"].classes_)
    rf_leave_p = float(rf_prob[rf_cls_list.index(lc)])

    avg_leave_p = (lr_leave_p + rf_leave_p) / 2
    return {
        "lr_leave": bool(lr_cls == lc),
        "lr_p": lr_leave_p,
        "rf_leave": bool(rf_cls == lc),
        "rf_p": rf_leave_p,
        "avg_p": avg_leave_p,
        "will_leave": avg_leave_p >= ENSEMBLE_THRESHOLD,
        "agree": lr_cls == rf_cls,
    }


def predict_batch(mi: Dict[str, Any], batch_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Batch prediction — requires all feature columns (see assess_batch_columns)."""
    check = assess_batch_columns(batch_df, mi["feat_cols"])
    if not check["can_predict"]:
        missing = ", ".join(check["missing"])
        raise ValueError(
            f"Batch CSV is missing required columns: {missing}. "
            "Predictions blocked to avoid unreliable default values."
        )

    feat_cols: List[str] = mi["feat_cols"]
    lc = mi["leave_class"]
    rows_list = [build_row(row.to_dict(), feat_cols) for _, row in batch_df.iterrows()]
    inp_df = pd.DataFrame(rows_list, columns=feat_cols)

    lr_cls_vec = mi["lr_pipe"].predict(inp_df)
    lr_prob_vec = mi["lr_pipe"].predict_proba(inp_df)
    lr_leave_idx = list(mi["lr_pipe"].named_steps["logisticregression"].classes_).index(
        lc
    )
    lr_leave_p_vec = lr_prob_vec[:, lr_leave_idx]

    rf_cls_vec = mi["rf_pipe"].predict(inp_df)
    rf_prob_vec = mi["rf_pipe"].predict_proba(inp_df)
    rf_leave_idx = list(
        mi["rf_pipe"].named_steps["randomforestclassifier"].classes_
    ).index(lc)
    rf_leave_p_vec = rf_prob_vec[:, rf_leave_idx]

    avg_leave_p_vec = (lr_leave_p_vec + rf_leave_p_vec) / 2
    both_agree_vec = lr_cls_vec == rf_cls_vec

    return [
        {
            "lr_leave": bool(lr_cls_vec[i] == lc),
            "lr_p": float(lr_leave_p_vec[i]),
            "rf_leave": bool(rf_cls_vec[i] == lc),
            "rf_p": float(rf_leave_p_vec[i]),
            "avg_p": float(avg_leave_p_vec[i]),
            "will_leave": avg_leave_p_vec[i] >= ENSEMBLE_THRESHOLD,
            "agree": bool(both_agree_vec[i]),
        }
        for i in range(len(batch_df))
    ]
