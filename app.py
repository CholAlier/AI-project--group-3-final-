from __future__ import annotations

import io
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from phoenix_ml import (
    CAT_FEATURES,
    assess_batch_columns,
    predict,
    predict_batch,
    train_models as _train_models,
)

# ── App config ──────────────────────────────────────────────────────────────
CSV_PATH = "HR-Employee-Attrition.csv"

st.set_page_config(
    page_title="Phoenix AI — Attrition Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme ───────────────────────────────────────────────────────────────────
T = {
    "bg": "#0d1f14",
    "bg2": "#0f2818",
    "surface": "rgba(255,255,255,0.055)",
    "border": "rgba(76,175,80,0.18)",
    "text": "#e8f0e8",
    "muted": "#6b9a6f",
    "gold": "#81c784",
    "teal": "#66bb6a",
    "coral": "#81c784",
    "blue": "#66bb6a",
    "green": "#68d391",
    "warn": "#81c784",
    "danger": "#fc8181",
    "success": "#68d391",
}


# ── CSS ──────────────────────────────────────────────────────────────────────
def inject_css() -> None:
    st.markdown(
        f"""
  <style>
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

  html, body, [class*="css"] {{ font-family:'Plus Jakarta Sans',sans-serif; }}
  h1,h2,h3,h4 {{ font-family:'Space Grotesk',sans-serif; }}
  #MainMenu, footer, header {{ visibility:hidden; }}
.block-container {{ padding-top:1.2rem; max-width:1300px; }}

.stApp {{
    background:
      radial-gradient(ellipse at 8% -5%, rgba(129,199,132,0.16) 0%, transparent 50%),
      radial-gradient(ellipse at 92% 8%, rgba(102,187,106,0.13) 0%, transparent 50%),
      radial-gradient(ellipse at 50% 95%, rgba(129,199,132,0.08) 0%, transparent 45%),
      linear-gradient(160deg, #0d1f14 0%, #0f2818 50%, #122819 100%);
    color:{T["text"]};
  }}
  p,span,label,li,div {{ color:{T["text"]}; }}

  /* ── top bar ── */
.topbar {{
    display:flex; align-items:center; justify-content:space-between;
    padding:.8rem 1.6rem; border-radius:18px;
    background:rgba(255,255,255,0.04);
    border:1px solid rgba(129,199,132,0.2);
    backdrop-filter:blur(24px); margin-bottom:1.6rem;
    box-shadow:0 4px 24px rgba(0,0,0,0.3);
  }}
.brand {{ font-family:'Space Grotesk',sans-serif; font-weight:700;
    font-size:1.15rem; display:flex; align-items:center; gap:.55rem; }}
.brand-icon {{ width:32px; height:32px; border-radius:9px;
    background:linear-gradient(135deg,{T["gold"]},{T["teal"]});
    display:flex; align-items:center; justify-content:center;
    font-size:1rem; box-shadow:0 0 16px rgba(129,199,132,0.45); }}
.status-dot {{ font-size:.74rem; font-weight:700; padding:.28rem .85rem;
    border-radius:999px; background:rgba(104,211,145,0.12);
    border:1px solid rgba(104,211,145,0.4); color:{T["green"]};
    letter-spacing:.04em; animation:pulse 2.5s ease-in-out infinite; }}

  /* ── hero ── */
.hero {{
    border-radius:28px; padding:3.4rem 2.8rem; margin-bottom:2rem;
    background:linear-gradient(135deg,#1b5e20 0%,#2e7d32 45%,#1b5e20 100%);
    background-size:200% 200%;
    animation:hshift 12s ease-in-out infinite, fadeUp .6s ease-out;
    border:1px solid rgba(129,199,132,0.25);
    box-shadow:0 32px 80px -20px rgba(102,187,106,0.35),
          0 0 0 1px rgba(255,255,255,0.04);
    position:relative; overflow:hidden;
  }}
.hero::before {{
    content:""; position:absolute; inset:0;
    background:
      radial-gradient(circle at 85% 50%, rgba(246,173,85,0.1) 0%, transparent 55%),
      radial-gradient(circle at 15% 80%, rgba(56,178,172,0.08) 0%, transparent 45%);
    pointer-events:none;
  }}
.hero::after {{
    content:""; position:absolute; inset:0;
    background-image:radial-gradient(circle,rgba(255,255,255,0.25) 1px,transparent 1px);
    background-size:28px 28px; opacity:0.06; pointer-events:none;
  }}
  @keyframes hshift {{ 0%,100%{{background-position:0% 50%;}} 50%{{background-position:100% 50%;}} }}
  @keyframes fadeUp {{ from{{opacity:0;transform:translateY(16px);}} to{{opacity:1;transform:translateY(0);}} }}
  @keyframes popIn {{ from{{opacity:0;transform:scale(.88);}} to{{opacity:1;transform:scale(1);}} }}
  @keyframes spin {{ to{{transform:rotate(360deg);}} }}
  @keyframes pulse {{ 0%,100%{{opacity:1;}} 50%{{opacity:.55;}} }}
  @keyframes float {{ 0%,100%{{transform:translateY(0);}} 50%{{transform:translateY(-5px);}} }}
.hero-label {{
    display:inline-flex; align-items:center; gap:.35rem;
    font-size:.72rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase;
    background:rgba(255,255,255,0.1); border:1px solid rgba(255,255,255,0.22);
    padding:.32rem .9rem; border-radius:999px; color:#fff;
    margin-bottom:1rem; backdrop-filter:blur(8px);
  }}
.hero-title {{
    font-family:'Space Grotesk',sans-serif; font-size:2.8rem; font-weight:700;
    color:#fff; line-height:1.12; letter-spacing:-.03em; margin-bottom:.75rem;
    text-shadow:0 4px 24px rgba(0,0,0,0.3);
  }}
.hero-sub {{ font-size:1.05rem; color:rgba(255,255,255,0.88); max-width:600px; line-height:1.7; }}
.hero-chips {{ display:flex; gap:.5rem; flex-wrap:wrap; margin-top:1.4rem; }}
.chip {{
    background:rgba(255,255,255,0.12); border:1px solid rgba(255,255,255,0.22);
    padding:.3rem .85rem; border-radius:999px; font-size:.76rem;
    font-weight:700; color:#fff; backdrop-filter:blur(6px);
    animation:float 4s ease-in-out infinite;
  }}
.chip:nth-child(2){{animation-delay:.5s;}}
.chip:nth-child(3){{animation-delay:1s;}}
.chip:nth-child(4){{animation-delay:1.5s;}}
  </style>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
  <style>
  /* ── glass card ── */
.card {{
    background:rgba(255,255,255,0.05);
    border:1px solid rgba(99,179,237,0.18); border-radius:20px;
    padding:1.4rem 1.5rem; backdrop-filter:blur(16px);
    box-shadow:0 8px 32px rgba(0,0,0,0.22);
    transition:all.25s cubic-bezier(.4,0,.2,1);
  }}
.card:hover {{
    transform:translateY(-3px); border-color:rgba(246,173,85,0.4);
    box-shadow:0 16px 48px rgba(246,173,85,0.12),0 0 0 1px rgba(246,173,85,0.15);
  }}

  /* ── kpi ── */
.kpi {{
    background:rgba(255,255,255,0.05); border:1px solid rgba(99,179,237,0.18);
    border-radius:18px; padding:1.2rem 1.3rem; text-align:center;
    transition:all.25s; position:relative; overflow:hidden;
  }}
.kpi::before {{
    content:""; position:absolute; top:0; left:0; right:0; height:2px;
    background:linear-gradient(90deg,{T["gold"]},{T["teal"]});
    opacity:0; transition:opacity.25s;
  }}
.kpi:hover {{ transform:translateY(-4px); box-shadow:0 12px 32px rgba(246,173,85,0.15); }}
.kpi:hover::before {{ opacity:1; }}
.kpi-val {{
    font-family:'Space Grotesk',sans-serif; font-size:1.7rem; font-weight:700;
    background:linear-gradient(135deg,{T["gold"]},{T["teal"]});
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
  }}
.kpi-lbl {{ font-size:.72rem; color:{T["muted"]}; text-transform:uppercase;
    letter-spacing:.07em; margin-top:.2rem; }}

  /* ── section heading ── */
.sec-head {{
    font-family:'Space Grotesk',sans-serif; font-size:1.3rem; font-weight:700;
    margin:.4rem 0 1.1rem 0; display:flex; align-items:center; gap:.5rem;
    padding-bottom:.6rem; border-bottom:1px solid rgba(99,179,237,0.12);
  }}

  /* ── result cards ── */
.result {{
    border-radius:28px; padding:2.6rem 2.2rem; text-align:center;
    margin-bottom:1.6rem; animation:popIn.5s cubic-bezier(.26,1.36,.44,1);
    position:relative; overflow:hidden;
  }}
.result::after {{
    content:""; position:absolute; inset:0;
    background-image:radial-gradient(circle,rgba(255,255,255,0.3) 1px,transparent 1px);
    background-size:24px 24px; opacity:0.07; pointer-events:none;
  }}
.result.high {{
    background:linear-gradient(135deg,#c0392b 0%,#e74c3c 40%,#e67e22 100%);
    box-shadow:0 32px 80px -20px rgba(252,129,129,0.5);
  }}
.result.low {{
    background:linear-gradient(135deg,#0e6655 0%,#1abc9c 45%,#0b7dda 100%);
    box-shadow:0 32px 80px -20px rgba(104,211,145,0.5);
  }}
.result-title {{ font-size:.82rem; font-weight:700; letter-spacing:.14em;
    text-transform:uppercase; color:rgba(255,255,255,.9); }}
.result-pct {{ font-family:'Space Grotesk',sans-serif; font-size:3rem; font-weight:700;
    color:#fff; margin:.4rem 0; text-shadow:0 4px 16px rgba(0,0,0,0.3); }}
.result-sub {{ color:rgba(255,255,255,.88); font-size:.95rem; }}

  /* ── rec cards ── */
.rec {{
    border-left:3px solid {T["teal"]}; background:rgba(56,178,172,0.06);
    border-radius:0 12px 12px 0; padding:.9rem 1.1rem; margin-bottom:.6rem;
    transition:transform.2s;
  }}
.rec:hover {{ transform:translateX(3px); }}
.rec.warn {{ border-left-color:{T["warn"]}; background:rgba(246,173,85,0.06); }}
.rec.danger {{ border-left-color:{T["danger"]}; background:rgba(252,129,129,0.06); }}
.rec.good {{ border-left-color:{T["success"]}; background:rgba(104,211,145,0.06); }}

  /* ── button ── */
  div.stButton > button {{
    background:linear-gradient(120deg,{T["gold"]},{T["teal"]})!important;
    color:#06111f!important; font-weight:800; border:none;
    border-radius:14px; padding:.75rem 1.6rem;
    box-shadow:0 12px 32px rgba(246,173,85,0.35);
    transition:all.22s cubic-bezier(.4,0,.2,1); width:100%;
    font-size:1rem; letter-spacing:.01em;
  }}
  div.stButton > button:hover {{
    transform:translateY(-3px) scale(1.01);
    box-shadow:0 18px 44px rgba(56,178,172,0.45);
  }}
  div.stButton > button:active {{ transform:translateY(0) scale(.98); }}
  div.stButton > button p {{ color:#06111f!important; }}

  /* ── model badges ── */
.mb-lr {{ display:inline-flex; align-items:center; gap:.4rem;
    padding:.35rem .9rem; border-radius:999px; font-size:.8rem; font-weight:700; margin:.15rem;
    background:rgba(99,179,237,0.12); color:{T["blue"]}; border:1px solid rgba(99,179,237,0.35); }}
.mb-rf {{ display:inline-flex; align-items:center; gap:.4rem;
    padding:.35rem .9rem; border-radius:999px; font-size:.8rem; font-weight:700; margin:.15rem;
    background:rgba(104,211,145,0.12); color:{T["green"]}; border:1px solid rgba(104,211,145,0.35); }}

  /* ── tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    gap:5px; background:rgba(255,255,255,0.03); border-radius:14px; padding:4px;
    border:1px solid rgba(99,179,237,0.12);
  }}
.stTabs [data-baseweb="tab"] {{
    border-radius:10px; padding:.48rem 1rem; background:transparent;
    border:none; color:{T["muted"]}; font-weight:600; transition:all.2s;
  }}
.stTabs [data-baseweb="tab"]:hover {{ background:rgba(255,255,255,0.06); color:{T["text"]}; }}
.stTabs [aria-selected="true"] {{
    background:linear-gradient(120deg,{T["gold"]},{T["teal"]})!important;
    color:#06111f!important; font-weight:800;
    box-shadow:0 4px 14px rgba(246,173,85,0.3);
  }}

  /* ── sidebar ── */
  section[data-testid="stSidebar"] {{
    background:linear-gradient(180deg,#091428,#060d1f);
    border-right:1px solid rgba(99,179,237,0.12);
  }}

  /* ── divider ── */
.div {{ height:1px; border:none; margin:2rem 0;
    background:linear-gradient(90deg,transparent,rgba(99,179,237,0.25),transparent); }}

  /* ── scrollbar ── */
::-webkit-scrollbar {{ width:6px; height:6px; }}
::-webkit-scrollbar-track {{ background:transparent; }}
::-webkit-scrollbar-thumb {{ background:rgba(99,179,237,0.25); border-radius:999px; }}
::-webkit-scrollbar-thumb:hover {{ background:rgba(246,173,85,0.4); }}

  /* ── loading ── */
.orb {{ width:68px; height:68px; border-radius:50%; margin:0 auto 1.2rem;
    background:conic-gradient(from 0deg,{T["gold"]},{T["teal"]},{T["coral"]},{T["gold"]});
    animation:spin 1.1s linear infinite; box-shadow:0 0 48px rgba(246,173,85,0.45); }}
.load-txt {{ font-size:1rem; font-weight:600; text-align:center; color:{T["text"]}; }}
.ptrack {{ width:100%; height:6px; border-radius:999px;
    background:rgba(99,179,237,0.15); overflow:hidden; }}
.pfill {{ height:100%; border-radius:999px;
    background:linear-gradient(90deg,{T["gold"]},{T["teal"]}); transition:width.35s ease; }}

  @media(max-width:768px){{
.hero-title{{font-size:1.9rem;}}
.result-pct{{font-size:2.1rem;}}
  }}
  </style>""",
        unsafe_allow_html=True,
    )


# ── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def train_models(_df: pd.DataFrame) -> Dict[str, Any]:
    return _train_models(_df)


# ── Rule-based insights ───────────────────────────────────────────────────────
def get_insights(
    df: pd.DataFrame, raw: Dict[str, Any], pred: Dict[str, Any]
) -> Dict[str, Any]:
    dept_avg = df.loc[df["Department"] == raw["Department"], "MonthlyIncome"].mean()
    notes: List[Tuple[str, str]] = []

    if raw["OverTime"] == "Yes":
        notes.append(
            (
                "danger",
                "Regularly works overtime — one of the strongest attrition signals.",
            )
        )
    if raw["MonthlyIncome"] < dept_avg:
        notes.append(
            (
                "warn",
                f"Income (${raw['MonthlyIncome']:,}) is below {raw['Department']} avg (${dept_avg:,.0f}).",
            )
        )
    if raw["YearsAtCompany"] <= 2:
        notes.append(
            (
                "warn",
                "Short tenure — employees in the first 2 years leave more frequently.",
            )
        )
    if raw["JobSatisfaction"] <= 2:
        notes.append(("danger", "Low job satisfaction reported."))
    if raw["EnvironmentSatisfaction"] <= 2:
        notes.append(("warn", "Low environment satisfaction reported."))
    if raw["WorkLifeBalance"] <= 2:
        notes.append(("warn", "Poor work-life balance."))
    if raw["YearsSinceLastPromotion"] >= 5:
        notes.append(
            ("warn", "No promotion in 5+ years — career stagnation is a key driver.")
        )
    if raw["NumCompaniesWorked"] >= 5:
        notes.append(
            ("warn", "Frequent job changes across companies in career history.")
        )
    if raw["BusinessTravel"] == "Travel_Frequently":
        notes.append(("warn", "Frequent business travel contributes to burnout."))
    if not notes:
        notes.append(
            ("good", "No major risk indicators found across key workforce signals.")
        )

    eng = float(
        np.clip(
            (
                np.mean(
                    [
                        raw["EnvironmentSatisfaction"],
                        raw["JobSatisfaction"],
                        raw["JobInvolvement"],
                        raw["WorkLifeBalance"],
                    ]
                )
                - 1
            )
            / 3
            * 100,
            0,
            100,
        )
    )
    stb = float(np.clip(raw["YearsAtCompany"] / 15 * 100, 0, 100))

    return {
        "notes": notes,
        "engagement": eng,
        "stability": stb,
        "dept_avg": dept_avg,
    }


# ── UI helpers ────────────────────────────────────────────────────────────────
LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Plus Jakarta Sans, sans-serif", color=T["text"]),
    margin=dict(l=16, r=16, t=40, b=16),
)

# The three scored models, in the same order the notebook reports them.
MODEL_NAMES = ["Logistic Regression", "Random Forest", "Combined Prediction"]
# Blue / green / amber — distinct hues so the three ROC curves stay readable.
MODEL_COLORS = ["#63b3ed", "#68d391", "#f6ad55"]


def kpi(icon: str, val: str, label: str) -> str:
    icon_html = f'<div style="font-size:1.2rem">{icon}</div>' if icon else ""
    return (
        f'<div class="kpi">{icon_html}'
        f'<div class="kpi-val">{val}</div>'
        f'<div class="kpi-lbl">{label}</div></div>'
    )


def gauge(value: float, title: str, color: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": "%", "font": {"size": 24, "family": "Plus Jakarta Sans"}},
            title={"text": title, "font": {"size": 12}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": T["muted"]},
                "bar": {"color": color},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "rgba(104,211,145,0.12)"},
                    {"range": [40, 70], "color": "rgba(246,173,85,0.12)"},
                    {"range": [70, 100], "color": "rgba(252,129,129,0.12)"},
                ],
            },
        )
    )
    fig.update_layout(height=200, **LAYOUT)
    return fig


def render_topbar() -> None:
    st.markdown(
        """
  <div style="padding:1.4rem 0 0.4rem 0;">
    <div style="font-family:'Space Grotesk',sans-serif; font-size:2rem;
       font-weight:800; letter-spacing:-0.03em; line-height:1.1;
       background:linear-gradient(135deg,#f6ad55,#38b2ac);
       -webkit-background-clip:text; -webkit-text-fill-color:transparent;
       background-clip:text;">
      Phoenix AI
    </div>
  </div>
  <hr style="border:none; height:1px; margin:0.8rem 0 1.4rem 0;
    background:linear-gradient(90deg,transparent,rgba(99,179,237,0.25),transparent);">
  """,
        unsafe_allow_html=True,
    )


def render_sidebar(mi: Dict[str, Any], df: pd.DataFrame) -> None:
    with st.sidebar:
        st.markdown(
            """
    <div style="display:flex;align-items:center;gap:.6rem;padding:.8rem;
       border-radius:14px;background:rgba(255,255,255,0.05);
       border:1px solid rgba(99,179,237,0.15);margin-bottom:1rem;">
      <div style="width:32px;height:32px;border-radius:9px;
         background:linear-gradient(135deg,#f6ad55,#38b2ac);
         display:flex;align-items:center;justify-content:center;font-size:1rem;"></div>
      <div><b>Phoenix AI</b><br>
      <span style="font-size:.72rem;color:#6b8cba;">Attrition Intelligence</span></div>
    </div>""",
            unsafe_allow_html=True,
        )

        st.markdown("### Dataset Summary")
        attr_rate = (df["Attrition"] == "Yes").mean() * 100
        st.markdown(
            f"""
    <div class="card" style="padding:.9rem 1rem;">
      <b>Employees:</b> {len(df):,}<br>
      <b>Attrition Rate:</b> {attr_rate:.1f}%<br>
      <b>Features:</b> {len(mi["feat_cols"])}<br>
      <b>Train / Val / Test:</b>
      {mi["n_train"]:,} / {mi["n_val"]:,} / {mi["n_test"]:,}
    </div>""",
            unsafe_allow_html=True,
        )

        st.markdown("### Model Metrics")
        st.caption("Measured on the held-out test set.")
        st.markdown(
            f"""
    <div class="card" style="padding:.9rem 1rem;font-size:.84rem;line-height:1.9;">
      <b>Logistic Regression</b><br>
      Accuracy: <b>{mi["lr_acc"] * 100:.2f}%</b> &nbsp;·&nbsp;
      Precision: <b>{mi["lr_precision"] * 100:.2f}%</b> &nbsp;·&nbsp;
      Recall: <b>{mi["lr_recall"] * 100:.2f}%</b> &nbsp;·&nbsp;
      ROC-AUC: <b>{mi["lr_roc"]:.3f}</b><br>
      <b>Random Forest</b><br>
      Accuracy: <b>{mi["rf_acc"] * 100:.2f}%</b> &nbsp;·&nbsp;
      Precision: <b>{mi["rf_precision"] * 100:.2f}%</b> &nbsp;·&nbsp;
      Recall: <b>{mi["rf_recall"] * 100:.2f}%</b> &nbsp;·&nbsp;
      ROC-AUC: <b>{mi["rf_roc"]:.3f}</b><br>
      <b>Combined Prediction</b><br>
      Accuracy: <b>{mi["ens_acc"] * 100:.2f}%</b> &nbsp;·&nbsp;
      Precision: <b>{mi["ens_precision"] * 100:.2f}%</b> &nbsp;·&nbsp;
      Recall: <b>{mi["ens_recall"] * 100:.2f}%</b> &nbsp;·&nbsp;
      ROC-AUC: <b>{mi["ens_roc"]:.3f}</b>
    </div>""",
            unsafe_allow_html=True,
        )

        st.markdown("### Prediction History")
        if st.session_state.get("history"):
            for h in reversed(st.session_state.history[-5:]):
                label = "High" if h["leave"] else "Low"
                st.markdown(
                    f"""
        <div class="card" style="padding:.7rem .9rem;margin-bottom:.4rem;">
          {label}: <b>{h["risk"]:.0f}% risk</b> — {h["role"]}<br>
          <span style="font-size:.72rem;color:#6b8cba;">{h["dept"]} · {h["time"]}</span>
        </div>""",
                    unsafe_allow_html=True,
                )
            hist_df = pd.DataFrame(st.session_state.history)
            buf = io.StringIO()
            hist_df.to_csv(buf, index=False)
            st.download_button(
                "Export History CSV",
                buf.getvalue(),
                "phoenix_history.csv",
                "text/csv",
                use_container_width=True,
            )
        else:
            st.markdown(
                '<div class="card" style="padding:.7rem .9rem;color:#6b8cba;">No predictions yet.</div>',
                unsafe_allow_html=True,
            )

        if st.button("Reset Form"):
            for k in [k for k in st.session_state if k.startswith("f_")]:
                del st.session_state[k]
            st.session_state.result = None
            st.rerun()


def render_form(df: pd.DataFrame) -> Dict[str, Any]:
    st.markdown('<div class="sec-head">Employee Profile</div>', unsafe_allow_html=True)
    form_tabs = st.tabs(
        [
            "Employee",
            "Job",
            "Compensation",
            "Environment",
            "Performance",
            "Lifestyle",
        ]
    )
    inp: Dict[str, Any] = {}

    with form_tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            inp["Age"] = st.slider(
                "Age", 18, 60, st.session_state.get("f_Age", 30), key="f_Age"
            )
            inp["Gender"] = st.radio(
                "Gender", ["Male", "Female"], horizontal=True, key="f_Gender"
            )
        with c2:
            inp["MaritalStatus"] = st.selectbox(
                "Marital Status",
                ["Single", "Married", "Divorced"],
                key="f_MaritalStatus",
            )
            inp["DistanceFromHome"] = st.slider(
                "Distance From Home (km)",
                1,
                29,
                st.session_state.get("f_DistanceFromHome", 8),
                key="f_DistanceFromHome",
            )

    with form_tabs[1]:
        c1, c2 = st.columns(2)
        with c1:
            inp["Department"] = st.selectbox(
                "Department",
                ["Sales", "Research & Development", "Human Resources"],
                key="f_Department",
            )
            inp["JobRole"] = st.selectbox(
                "Job Role",
                [
                    "Sales Executive",
                    "Research Scientist",
                    "Laboratory Technician",
                    "Manufacturing Director",
                    "Healthcare Representative",
                    "Manager",
                    "Sales Representative",
                    "Research Director",
                    "Human Resources",
                ],
                key="f_JobRole",
            )
            inp["EducationField"] = st.selectbox(
                "Education Field",
                [
                    "Life Sciences",
                    "Other",
                    "Medical",
                    "Marketing",
                    "Technical Degree",
                    "Human Resources",
                ],
                key="f_EducationField",
            )
        with c2:
            inp["BusinessTravel"] = st.selectbox(
                "Business Travel",
                ["Travel_Rarely", "Travel_Frequently", "Non-Travel"],
                key="f_BusinessTravel",
            )
            inp["Education"] = st.select_slider(
                "Education Level",
                [1, 2, 3, 4, 5],
                value=st.session_state.get("f_Education", 3),
                key="f_Education",
                help="1=Below College · 5=Doctorate",
            )

    with form_tabs[2]:
        c1, c2 = st.columns(2)
        with c1:
            inp["MonthlyIncome"] = st.number_input(
                "Monthly Income ($)",
                1000,
                20000,
                st.session_state.get("f_MonthlyIncome", 5000),
                100,
                key="f_MonthlyIncome",
            )
            inp["DailyRate"] = st.number_input(
                "Daily Rate ($)",
                100,
                1500,
                st.session_state.get("f_DailyRate", 750),
                10,
                key="f_DailyRate",
            )
            inp["HourlyRate"] = st.number_input(
                "Hourly Rate ($)",
                30,
                100,
                st.session_state.get("f_HourlyRate", 60),
                1,
                key="f_HourlyRate",
            )
        with c2:
            inp["MonthlyRate"] = st.number_input(
                "Monthly Rate ($)",
                2000,
                27000,
                st.session_state.get("f_MonthlyRate", 13000),
                100,
                key="f_MonthlyRate",
            )
            inp["PercentSalaryHike"] = st.slider(
                "Salary Hike %",
                11,
                25,
                st.session_state.get("f_PercentSalaryHike", 14),
                key="f_PercentSalaryHike",
            )
            inp["StockOptionLevel"] = st.select_slider(
                "Stock Option Level",
                [0, 1, 2, 3],
                value=st.session_state.get("f_StockOptionLevel", 1),
                key="f_StockOptionLevel",
            )

    with form_tabs[3]:
        c1, c2 = st.columns(2)
        with c1:
            inp["EnvironmentSatisfaction"] = st.select_slider(
                "Environment Satisfaction",
                [1, 2, 3, 4],
                value=st.session_state.get("f_ES", 3),
                key="f_ES",
                help="1=Low · 4=Very High",
            )
            inp["JobInvolvement"] = st.select_slider(
                "Job Involvement",
                [1, 2, 3, 4],
                value=st.session_state.get("f_JI", 3),
                key="f_JI",
            )
        with c2:
            inp["WorkLifeBalance"] = st.select_slider(
                "Work-Life Balance",
                [1, 2, 3, 4],
                value=st.session_state.get("f_WLB", 3),
                key="f_WLB",
            )
            inp["OverTime"] = st.radio(
                "Works Overtime?", ["No", "Yes"], horizontal=True, key="f_OT"
            )

    with form_tabs[4]:
        c1, c2 = st.columns(2)
        with c1:
            inp["JobSatisfaction"] = st.select_slider(
                "Job Satisfaction",
                [1, 2, 3, 4],
                value=st.session_state.get("f_JS", 3),
                key="f_JS",
            )
            inp["RelationshipSatisfaction"] = st.select_slider(
                "Relationship Satisfaction",
                [1, 2, 3, 4],
                value=st.session_state.get("f_RS", 3),
                key="f_RS",
            )
        with c2:
            inp["YearsSinceLastPromotion"] = st.slider(
                "Years Since Last Promotion",
                0,
                15,
                st.session_state.get("f_YSLP", 2),
                key="f_YSLP",
            )
            inp["TrainingTimesLastYear"] = st.slider(
                "Training Sessions Last Year",
                0,
                6,
                st.session_state.get("f_TT", 3),
                key="f_TT",
            )

    with form_tabs[5]:
        c1, c2 = st.columns(2)
        with c1:
            inp["NumCompaniesWorked"] = st.slider(
                "Companies Worked At",
                0,
                9,
                st.session_state.get("f_NCW", 2),
                key="f_NCW",
            )
        with c2:
            inp["YearsAtCompany"] = st.slider(
                "Years At Company", 0, 40, st.session_state.get("f_YAC", 5), key="f_YAC"
            )

    with st.expander("Employee Reference"):
        inp["EmployeeNumber"] = st.number_input(
            "Employee Number", 1, 3000, st.session_state.get("f_EN", 1001), key="f_EN"
        )

    return inp


def run_prediction(mi: Dict[str, Any], df: pd.DataFrame, inp: Dict[str, Any]) -> None:
    msgs = [
        "Loading profile...",
        "Evaluating workforce signals...",
        "Running Random Forest...",
        "Running Logistic Regression...",
        "Combining model outputs...",
        "Done.",
    ]
    ph = st.empty()
    pb = st.empty()
    for i, m in enumerate(msgs):
        pct = int((i + 1) / len(msgs) * 100)
        ph.markdown(
            f'<div style="text-align:center;padding:2rem"><div class="orb"></div>'
            f'<div class="load-txt">{m}</div></div>',
            unsafe_allow_html=True,
        )
        pb.markdown(
            f'<div class="ptrack"><div class="pfill" style="width:{pct}%"></div></div>',
            unsafe_allow_html=True,
        )
        time.sleep(0.4)

    try:
        pred = predict(mi, inp)
    except Exception as e:
        ph.empty()
        pb.empty()
        st.error(f"Prediction error: {e}")
        return

    ph.empty()
    pb.empty()
    ins = get_insights(df, inp, pred)
    st.session_state.result = pred
    st.session_state.inputs = dict(inp)
    st.session_state.insight = ins
    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append(
        {
            "leave": pred["will_leave"],
            "risk": pred["avg_p"] * 100,
            "role": inp["JobRole"],
            "dept": inp["Department"],
            "time": datetime.now().strftime("%H:%M:%S"),
        }
    )


def render_results() -> None:
    pred = st.session_state.result
    inp = st.session_state.inputs
    ins = st.session_state.insight
    risk = pred["avg_p"] * 100
    conf = max(pred["avg_p"], 1 - pred["avg_p"]) * 100

    if pred["will_leave"]:
        st.markdown(
            f"""
    <div class="result high">
      <div class="result-title">High Attrition Risk</div>
      <div class="result-pct">{risk:.1f}% Probability</div>
      <div class="result-sub">Avg of LR ({pred["lr_p"] * 100:.1f}%) &amp;
      RF ({pred["rf_p"] * 100:.1f}%) · Confidence {conf:.1f}%
      {"· Both models agree" if pred["agree"] else "· Models disagree"}</div>
    </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
    <div class="result low">
      <div class="result-title">Low Attrition Risk</div>
      <div class="result-pct">{100 - risk:.1f}% Likely to Stay</div>
      <div class="result-sub">Avg of LR ({pred["lr_p"] * 100:.1f}%) &amp;
      RF ({pred["rf_p"] * 100:.1f}%) · Confidence {conf:.1f}%
      {"· Both models agree" if pred["agree"] else "· Models disagree"}</div>
    </div>""",
            unsafe_allow_html=True,
        )

    # Gauges
    g1, g2, g3 = st.columns(3)
    with g1:
        st.plotly_chart(
            gauge(risk, "Attrition Risk", T["coral"]),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with g2:
        st.plotly_chart(
            gauge(conf, "Confidence", T["blue"]),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with g3:
        st.plotly_chart(
            gauge(ins["engagement"], "Engagement", T["green"]),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    # Model comparison bar
    st.markdown('<div class="sec-head">Model Comparison</div>', unsafe_allow_html=True)
    comp = pd.DataFrame(
        {
            "Model": MODEL_NAMES,
            "Attrition Probability (%)": [
                pred["lr_p"] * 100,
                pred["rf_p"] * 100,
                pred["avg_p"] * 100,
            ],
        }
    )
    fig = px.bar(
        comp,
        x="Model",
        y="Attrition Probability (%)",
        color="Model",
        color_discrete_sequence=MODEL_COLORS,
        text_auto=".1f",
    )
    fig.update_layout(**LAYOUT, showlegend=False, height=280)
    fig.add_hline(
        y=50,
        line_dash="dash",
        line_color=T["coral"],
        annotation_text="Decision threshold (50%)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Risk signals
    st.markdown('<div class="sec-head">Risk Signals</div>', unsafe_allow_html=True)
    icons = {"danger": "", "warn": "", "good": ""}
    for sev, txt in ins["notes"]:
        st.markdown(
            f'<div class="rec {sev}">{icons[sev]} {txt}</div>', unsafe_allow_html=True
        )

    # Recommendations
    st.markdown(
        '<div class="sec-head">Recommended Actions</div>', unsafe_allow_html=True
    )
    if pred["will_leave"]:
        st.markdown(
            """
    <div class="rec danger"> <b>Urgent:</b> Schedule a retention conversation within 2 weeks.</div>
    <div class="rec warn"> <b>Review:</b> Benchmark compensation, address overtime, discuss promotion paths.</div>
    <div class="rec warn"> <b>Engage:</b> Assign a mentor or offer a development opportunity.</div>
    """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
    <div class="rec good"> <b>Continue:</b> Current engagement practices are working well.</div>
    <div class="rec good"> <b>Develop:</b> Consider this employee for leadership or mentorship roles.</div>
    """,
            unsafe_allow_html=True,
        )

    # Summary row
    st.markdown('<div class="sec-head">Quick Summary</div>', unsafe_allow_html=True)
    cols = st.columns(6)
    items = [
        ("", f"{inp['Age']} yrs", "Age"),
        ("", inp["Department"], "Dept"),
        ("", inp["JobRole"][:14], "Role"),
        ("", f"${inp['MonthlyIncome']:,}", "Income"),
        ("", f"{inp['YearsAtCompany']} yrs", "Tenure"),
        ("", inp["OverTime"], "OT"),
    ]
    for col, (ic, v, lb) in zip(cols, items):
        with col:
            st.markdown(kpi(ic, v, lb), unsafe_allow_html=True)

    # Export report
    st.markdown("<br>", unsafe_allow_html=True)
    report = (
        f"Phoenix AI — Attrition Risk Report\n"
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"Department: {inp['Department']} | Role: {inp['JobRole']}\n"
        f"Age: {inp['Age']} | Income: ${inp['MonthlyIncome']:,} | Tenure: {inp['YearsAtCompany']} yrs\n\n"
        f"Logistic Regression: {pred['lr_p'] * 100:.1f}% attrition probability\n"
        f"Random Forest:    {pred['rf_p'] * 100:.1f}% attrition probability\n"
        f"Combined Score:   {risk:.1f}%\n"
        f"Verdict: {'HIGH RISK — likely to leave' if pred['will_leave'] else 'LOW RISK — likely to stay'}\n"
        f"Models {'agree' if pred['agree'] else 'disagree'} on outcome.\n"
    )
    st.download_button(
        "Export Report (.txt)",
        report,
        "phoenix_report.txt",
        "text/plain",
        use_container_width=True,
    )


def render_batch(mi: Dict[str, Any]) -> None:
    st.markdown(
        '<div class="sec-head">📁 Batch Prediction — Upload Employee CSV</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is None:
        return

    try:
        batch_df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"❌ Could not read file: {e}")
        return

    batch_df = batch_df.rename(columns={c: str(c).strip() for c in batch_df.columns})

    quality = assess_batch_columns(batch_df, mi["feat_cols"])
    present_cols = quality["present"]
    missing = quality["missing"]
    missing_critical = quality["missing_critical"]

    with st.expander("📊 Data Quality Check", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", len(batch_df))
        with col2:
            st.metric("Columns Found", f"{len(present_cols)}/{len(mi['feat_cols'])}")
        with col3:
            st.metric("Missing Columns", len(missing))

        if missing:
            st.error(
                f"Missing {len(missing)} required column(s): "
                f"{', '.join(missing[:8])}{'...' if len(missing) > 8 else ''}"
            )
            if missing_critical:
                st.warning(
                    "Critical fields missing: "
                    + ", ".join(missing_critical)
                    + ". Predictions are blocked until the CSV includes all model features."
                )
            else:
                st.warning(
                    "Predictions are blocked when any feature column is missing "
                    "to avoid unreliable default values."
                )
        else:
            st.success("✅ All expected columns present!")

    if not quality["can_predict"]:
        return

    # Vectorized batch predictions
    try:
        with st.spinner("🔄 Processing predictions..."):
            predictions = predict_batch(mi, batch_df)
    except Exception as e:
        st.error(f"❌ Batch prediction failed: {e}")
        return

    # Format results
    results = []
    for i, (_, row) in enumerate(batch_df.iterrows()):
        try:
            p = predictions[i]
            results.append(
                {
                    "Employee #": row.get("EmployeeNumber", i + 1),
                    "Department": row.get("Department", "—"),
                    "Job Role": row.get("JobRole", "—"),
                    "LR Risk %": round(p["lr_p"] * 100, 1),
                    "RF Risk %": round(p["rf_p"] * 100, 1),
                    "Avg Risk %": round(p["avg_p"] * 100, 1),
                    "Verdict": "🔴 High Risk" if p["will_leave"] else "🟢 Low Risk",
                    "Models Agree": "✅" if p["agree"] else "⚠️",
                }
            )
        except Exception:
            results.append(
                {
                    "Employee #": row.get("EmployeeNumber", i + 1),
                    "Department": row.get("Department", "—"),
                    "Job Role": row.get("JobRole", "—"),
                    "Verdict": "Error",
                }
            )

    out = pd.DataFrame(results)
    st.success(f"✅ Successfully predicted {len(out)} employees")

    high = (out["Verdict"] == "🔴 High Risk").sum()
    low = len(out) - high
    b1, b2, b3 = st.columns(3)
    with b1:
        st.markdown(kpi("📋", str(len(out)), "Total Processed"), unsafe_allow_html=True)
    with b2:
        st.markdown(kpi("🔴", str(high), "High Risk"), unsafe_allow_html=True)
    with b3:
        st.markdown(kpi("🟢", str(low), "Low Risk"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(out, use_container_width=True)

    buf = io.StringIO()
    out.to_csv(buf, index=False)
    st.download_button(
        "⬇️ Download Results CSV",
        buf.getvalue(),
        "phoenix_batch_results.csv",
        "text/csv",
        use_container_width=True,
    )


def render_analytics(df: pd.DataFrame, mi: Dict[str, Any]) -> None:
    st.markdown(
        '<div class="sec-head">Phoenix AI Analytics</div>', unsafe_allow_html=True
    )

    ana_tabs = st.tabs(
        [
            "Overview",
            "Department",
            "Salary",
            "Age & Tenure",
            "Gender",
            "Travel & OT",
            "Satisfaction",
            "Correlation",
            "Feature Impact",
            "Model Report",
            "EDA",
            "Ethics & Fairness",
        ]
    )

    colors = [T["green"], T["coral"]]

    with ana_tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            cnt = df["Attrition"].value_counts()
            fig = px.pie(
                values=cnt.values,
                names=cnt.index,
                hole=0.55,
                color_discrete_sequence=colors,
                title="Overall Attrition Split",
            )
            fig.update_layout(**LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            role_attr = (
                df.groupby("JobRole")["Attrition"]
                .apply(lambda x: (x == "Yes").mean() * 100)
                .reset_index()
            )
            role_attr.columns = ["Job Role", "Attrition Rate (%)"]
            role_attr = role_attr.sort_values("Attrition Rate (%)", ascending=True)
            fig = px.bar(
                role_attr,
                x="Attrition Rate (%)",
                y="Job Role",
                orientation="h",
                color="Attrition Rate (%)",
                color_continuous_scale="OrRd",
                title="Attrition Rate by Job Role",
            )
            fig.update_layout(**LAYOUT, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    with ana_tabs[1]:
        d = df.groupby(["Department", "Attrition"]).size().reset_index(name="Count")
        fig = px.bar(
            d,
            x="Department",
            y="Count",
            color="Attrition",
            barmode="group",
            color_discrete_sequence=colors,
            title="Attrition by Department",
        )
        fig.update_layout(**LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with ana_tabs[2]:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(
                df,
                x="MonthlyIncome",
                nbins=40,
                color="Attrition",
                color_discrete_sequence=colors,
                title="Income Distribution",
                barmode="overlay",
                opacity=0.75,
            )
            fig.update_layout(**LAYOUT, bargap=0.04)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.box(
                df,
                x="Department",
                y="MonthlyIncome",
                color="Attrition",
                color_discrete_sequence=colors,
                title="Income by Dept & Attrition",
            )
            fig.update_layout(**LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

    with ana_tabs[3]:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(
                df,
                x="Age",
                nbins=30,
                color="Attrition",
                color_discrete_sequence=colors,
                title="Age Distribution",
                barmode="overlay",
                opacity=0.75,
            )
            fig.update_layout(**LAYOUT, bargap=0.04)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.histogram(
                df,
                x="YearsAtCompany",
                nbins=20,
                color="Attrition",
                color_discrete_sequence=colors,
                title="Tenure Distribution",
                barmode="overlay",
                opacity=0.75,
            )
            fig.update_layout(**LAYOUT, bargap=0.04)
            st.plotly_chart(fig, use_container_width=True)

    with ana_tabs[4]:
        g = df.groupby(["Gender", "Attrition"]).size().reset_index(name="Count")
        fig = px.bar(
            g,
            x="Gender",
            y="Count",
            color="Attrition",
            barmode="group",
            color_discrete_sequence=colors,
            title="Attrition by Gender",
        )
        fig.update_layout(**LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with ana_tabs[5]:
        c1, c2 = st.columns(2)
        with c1:
            t = (
                df.groupby(["BusinessTravel", "Attrition"])
                .size()
                .reset_index(name="Count")
            )
            fig = px.bar(
                t,
                x="BusinessTravel",
                y="Count",
                color="Attrition",
                barmode="group",
                color_discrete_sequence=colors,
                title="Business Travel vs Attrition",
            )
            fig.update_layout(**LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            ot = df.groupby(["OverTime", "Attrition"]).size().reset_index(name="Count")
            fig = px.bar(
                ot,
                x="OverTime",
                y="Count",
                color="Attrition",
                barmode="group",
                color_discrete_sequence=colors,
                title="Overtime vs Attrition",
            )
            fig.update_layout(**LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

    with ana_tabs[6]:
        c1, c2 = st.columns(2)
        with c1:
            js = (
                df.groupby(["JobSatisfaction", "Attrition"])
                .size()
                .reset_index(name="Count")
            )
            fig = px.bar(
                js,
                x="JobSatisfaction",
                y="Count",
                color="Attrition",
                barmode="group",
                color_discrete_sequence=colors,
                title="Job Satisfaction",
            )
            fig.update_layout(**LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            wl = (
                df.groupby(["WorkLifeBalance", "Attrition"])
                .size()
                .reset_index(name="Count")
            )
            fig = px.bar(
                wl,
                x="WorkLifeBalance",
                y="Count",
                color="Attrition",
                barmode="group",
                color_discrete_sequence=colors,
                title="Work-Life Balance",
            )
            fig.update_layout(**LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

    with ana_tabs[7]:
        num_df = df.select_dtypes(include=["float", "int"])
        corr = num_df.corr()
        fig = px.imshow(
            corr,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Correlation Heatmap",
        )
        fig.update_layout(**LAYOUT, height=620)
        st.plotly_chart(fig, use_container_width=True)

    with ana_tabs[8]:
        fi = mi.get("feat_imp")
        if fi is not None and not fi.empty:
            fig = px.bar(
                fi,
                x="importance",
                y="feature",
                orientation="h",
                color="importance",
                color_continuous_scale="Teal",
                title="Top 15 Feature Importances (Random Forest)",
            )
            fig.update_layout(
                **LAYOUT,
                coloraxis_showscale=False,
                yaxis=dict(categoryorder="total ascending"),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Derived from Random Forest feature importances. "
                "Categorical features are aggregated across one-hot columns."
            )
        else:
            st.info("Feature importance unavailable.")

    with ana_tabs[9]:
        st.markdown("#### Full Model Performance Report")
        st.markdown(
            f"""
    <div class="card" style="margin-bottom:1rem;">
      The dataset is split <b>70% training</b> / <b>15% validation</b> /
      <b>15% testing</b> ({mi["n_train"]:,} / {mi["n_val"]:,} / {mi["n_test"]:,} rows,
      stratified on Attrition).<br>
      The validation set compares the models; the test set gives the final
      unbiased evaluation. <b>Combined Prediction</b> averages both models'
      attrition probabilities and applies a 50% threshold — this is the score
      the Predict tab reports.
    </div>""",
            unsafe_allow_html=True,
        )

        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.markdown(
                f"""
      <div class="card">
        <b>Logistic Regression</b><br><br>
        Accuracy &nbsp;: <b>{mi["lr_acc"] * 100:.2f}%</b><br>
        Precision: <b>{mi["lr_precision"] * 100:.2f}%</b><br>
        Recall &nbsp;&nbsp;&nbsp;: <b>{mi["lr_recall"] * 100:.2f}%</b><br>
        ROC-AUC &nbsp;: <b>{mi["lr_roc"]:.4f}</b><br><br>
        Type: Linear classifier<br>
        Encoding: One-Hot → StandardScaler<br>
        Max iter: 1000 · Random state: 42
      </div>""",
                unsafe_allow_html=True,
            )
        with mc2:
            st.markdown(
                f"""
      <div class="card">
        <b>Random Forest</b><br><br>
        Accuracy &nbsp;: <b>{mi["rf_acc"] * 100:.2f}%</b><br>
        Precision: <b>{mi["rf_precision"] * 100:.2f}%</b><br>
        Recall &nbsp;&nbsp;&nbsp;: <b>{mi["rf_recall"] * 100:.2f}%</b><br>
        ROC-AUC &nbsp;: <b>{mi["rf_roc"]:.4f}</b><br><br>
        Type: Ensemble (150 trees)<br>
        Encoding: One-Hot (no scaling needed)<br>
        Feature Importance: Yes · Random state: 42
      </div>""",
                unsafe_allow_html=True,
            )
        with mc3:
            st.markdown(
                f"""
      <div class="card">
        <b>Combined Prediction</b><br><br>
        Accuracy &nbsp;: <b>{mi["ens_acc"] * 100:.2f}%</b><br>
        Precision: <b>{mi["ens_precision"] * 100:.2f}%</b><br>
        Recall &nbsp;&nbsp;&nbsp;: <b>{mi["ens_recall"] * 100:.2f}%</b><br>
        ROC-AUC &nbsp;: <b>{mi["ens_roc"]:.4f}</b><br><br>
        Type: Averaged probabilities<br>
        Threshold: 50% → Leave<br>
        Used by: Predict &amp; Batch Predict
      </div>""",
                unsafe_allow_html=True,
            )

        # ── Validation set (model selection stage) ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Validation Set Performance")
        st.caption(
            f"Model selection stage — {mi['n_val']:,} held-out rows never seen during training."
        )
        val_df = pd.DataFrame(
            {
                "Model": MODEL_NAMES,
                "Accuracy": [
                    mi["lr_val"]["acc"],
                    mi["rf_val"]["acc"],
                    mi["ens_val"]["acc"],
                ],
                "Precision": [
                    mi["lr_val"]["precision"],
                    mi["rf_val"]["precision"],
                    mi["ens_val"]["precision"],
                ],
                "Recall": [
                    mi["lr_val"]["recall"],
                    mi["rf_val"]["recall"],
                    mi["ens_val"]["recall"],
                ],
                "ROC-AUC": [
                    mi["lr_val"]["roc"],
                    mi["rf_val"]["roc"],
                    mi["ens_val"]["roc"],
                ],
            }
        )
        st.dataframe(val_df.round(4), use_container_width=True, hide_index=True)

        # ── Test set comparison ──
        st.markdown("#### Final Test Set Performance")
        metrics_df = pd.DataFrame(
            {
                "Model": MODEL_NAMES,
                "Accuracy (%)": [
                    mi["lr_acc"] * 100,
                    mi["rf_acc"] * 100,
                    mi["ens_acc"] * 100,
                ],
                "Precision (%)": [
                    mi["lr_precision"] * 100,
                    mi["rf_precision"] * 100,
                    mi["ens_precision"] * 100,
                ],
                "Recall (%)": [
                    mi["lr_recall"] * 100,
                    mi["rf_recall"] * 100,
                    mi["ens_recall"] * 100,
                ],
                "ROC-AUC": [mi["lr_roc"], mi["rf_roc"], mi["ens_roc"]],
            }
        )
        st.dataframe(metrics_df.round(4), use_container_width=True, hide_index=True)

        for metric in ["Accuracy (%)", "Precision (%)", "Recall (%)", "ROC-AUC"]:
            fig = px.bar(
                metrics_df,
                x="Model",
                y=metric,
                color="Model",
                color_discrete_sequence=MODEL_COLORS,
                text_auto=".2f",
                title=f"{metric} Comparison",
            )
            fig.update_layout(**LAYOUT, showlegend=False, height=260)
            st.plotly_chart(fig, use_container_width=True)

        # ── ROC curves ──
        st.markdown("#### ROC Curves — Test Set")
        roc_fig = go.Figure()
        for proba, name, colr in zip(
            [mi["lr_proba"], mi["rf_proba"], mi["ens_proba"]],
            MODEL_NAMES,
            MODEL_COLORS,
        ):
            fpr, tpr, _ = roc_curve(mi["y_test_bin"], proba)
            auc = roc_auc_score(mi["y_test_bin"], proba)
            roc_fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"{name} (AUC={auc:.3f})",
                    line=dict(color=colr, width=2),
                )
            )
        roc_fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random Baseline",
                line=dict(color=T["muted"], width=1, dash="dash"),
            )
        )
        roc_fig.update_layout(
            **LAYOUT,
            height=420,
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            title="ROC Curve Comparison — Final Test Set",
            legend=dict(x=0.55, y=0.08, bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(roc_fig, use_container_width=True)

        # Confusion matrices side by side
        st.markdown("#### Confusion Matrices — Test Set")
        cm_cols = st.columns(3)
        for col, pred_arr, name in zip(
            cm_cols,
            [mi["lr_pred"], mi["rf_pred"], mi["ens_pred"]],
            MODEL_NAMES,
        ):
            with col:
                cm = confusion_matrix(mi["y_test"], pred_arr)
                fig = px.imshow(
                    cm,
                    text_auto=True,
                    x=["Predicted Stay", "Predicted Leave"],
                    y=["Actual Stay", "Actual Leave"],
                    color_continuous_scale="Blues",
                    title=f"{name} — Confusion Matrix",
                )
                fig.update_layout(**LAYOUT, height=300, coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)

    with ana_tabs[10]:
        st.markdown("#### Exploratory Data Analysis (EDA)")
        st.markdown(
            """
    <div class="card" style="margin-bottom:1rem;">
      Summary statistics and distribution of key features in the IBM HR dataset.
    </div>""",
            unsafe_allow_html=True,
        )

        e1, e2 = st.columns(2)
        with e1:
            # Class balance
            cnt = df["Attrition"].value_counts()
            fig = px.bar(
                x=cnt.index,
                y=cnt.values,
                color=cnt.index,
                color_discrete_sequence=[T["green"], T["coral"]],
                title="Class Balance (Attrition Yes vs No)",
                labels={"x": "Attrition", "y": "Count"},
            )
            fig.update_layout(**LAYOUT, showlegend=False, height=280)
            st.plotly_chart(fig, use_container_width=True)
        with e2:
            # Education field distribution
            ed = df["EducationField"].value_counts().reset_index()
            ed.columns = ["Field", "Count"]
            fig = px.bar(
                ed,
                x="Count",
                y="Field",
                orientation="h",
                color="Count",
                color_continuous_scale="Teal",
                title="Education Field Distribution",
            )
            fig.update_layout(
                **LAYOUT,
                coloraxis_showscale=False,
                yaxis=dict(categoryorder="total ascending"),
                height=280,
            )
            st.plotly_chart(fig, use_container_width=True)

        e3, e4 = st.columns(2)
        with e3:
            fig = px.box(
                df,
                x="Attrition",
                y="Age",
                color="Attrition",
                color_discrete_sequence=[T["green"], T["coral"]],
                title="Age Distribution by Attrition",
            )
            fig.update_layout(**LAYOUT, height=280, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with e4:
            fig = px.box(
                df,
                x="Attrition",
                y="MonthlyIncome",
                color="Attrition",
                color_discrete_sequence=[T["green"], T["coral"]],
                title="Monthly Income by Attrition",
            )
            fig.update_layout(**LAYOUT, height=280, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Dataset Statistics")
        st.dataframe(df.describe().round(2), use_container_width=True)

    with ana_tabs[11]:
        st.markdown("#### AI Ethics & Fairness Audit")
        st.markdown(
            """
    <div class="card" style="margin-bottom:1rem;">
      Checking for potential demographic bias in attrition predictions across
      <b>Age groups</b> and <b>Gender</b> — as required by the project ethics review.
    </div>""",
            unsafe_allow_html=True,
        )

        # Gender fairness
        st.markdown("##### Gender — Attrition Rate")
        gender_rate = (
            df.groupby("Gender")["Attrition"]
            .apply(lambda x: (x == "Yes").mean() * 100)
            .reset_index()
        )
        gender_rate.columns = ["Gender", "Attrition Rate (%)"]
        fig = px.bar(
            gender_rate,
            x="Gender",
            y="Attrition Rate (%)",
            color="Gender",
            color_discrete_sequence=[T["blue"], T["coral"]],
            text_auto=".1f",
            title="Attrition Rate by Gender",
        )
        fig.update_layout(**LAYOUT, showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)

        gap = abs(gender_rate["Attrition Rate (%)"].diff().iloc[-1])
        if gap > 5:
            st.markdown(
                f'<div class="rec warn"> Gender gap of {gap:.1f}% detected. '
                f"This may indicate bias in outcomes across gender groups.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="rec good"> Gender attrition gap is {gap:.1f}% — '
                f"within acceptable range.</div>",
                unsafe_allow_html=True,
            )

        # Age group fairness
        st.markdown("##### Age Groups — Attrition Rate")
        df_age = df.copy()
        df_age["Age Group"] = pd.cut(
            df_age["Age"],
            bins=[18, 25, 35, 45, 60],
            labels=["18-25", "26-35", "36-45", "46-60"],
        )
        age_rate = (
            df_age.groupby("Age Group", observed=True)["Attrition"]
            .apply(lambda x: (x == "Yes").mean() * 100)
            .reset_index()
        )
        age_rate.columns = ["Age Group", "Attrition Rate (%)"]
        fig = px.bar(
            age_rate,
            x="Age Group",
            y="Attrition Rate (%)",
            color="Age Group",
            color_discrete_sequence=[T["teal"], T["blue"], T["gold"], T["coral"]],
            text_auto=".1f",
            title="Attrition Rate by Age Group",
        )
        fig.update_layout(**LAYOUT, showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)

        highest = age_rate.loc[age_rate["Attrition Rate (%)"].idxmax(), "Age Group"]
        st.markdown(
            f'<div class="rec warn"> Highest attrition rate observed in the '
            f"<b>{highest}</b> age group. HR should investigate retention strategies "
            f"targeting this demographic.</div>",
            unsafe_allow_html=True,
        )

        # Department fairness
        st.markdown("##### Department — Attrition Rate")
        dept_rate = (
            df.groupby("Department")["Attrition"]
            .apply(lambda x: (x == "Yes").mean() * 100)
            .reset_index()
        )
        dept_rate.columns = ["Department", "Attrition Rate (%)"]
        fig = px.bar(
            dept_rate,
            x="Department",
            y="Attrition Rate (%)",
            color="Department",
            color_discrete_sequence=[T["teal"], T["gold"], T["coral"]],
            text_auto=".1f",
            title="Attrition Rate by Department",
        )
        fig.update_layout(**LAYOUT, showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            """
    <div class="card" style="margin-top:.5rem;">
      <b>Ethics Summary</b><br><br>
       <b>Gender and Age are used as predictor features</b>, so the models can
      learn demographic patterns directly — predictions must not be the sole basis
      for any HR decision.<br>
       Attrition rates are reported separately per demographic group above so
      disparities are visible.<br>
       Significant gaps in attrition rates across groups should be reviewed by HR.<br>
       Regular re-audits recommended as workforce demographics change.
    </div>""",
            unsafe_allow_html=True,
        )


# ── Session state ─────────────────────────────────────────────────────────────
def init_state() -> None:
    for key, val in [
        ("result", None),
        ("inputs", None),
        ("insight", None),
        ("history", []),
    ]:
        if key not in st.session_state:
            st.session_state[key] = val


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    try:
        raw_df = load_data(CSV_PATH)
    except FileNotFoundError:
        st.error(f"Dataset '{CSV_PATH}' not found. Place it next to app.py.")
        st.stop()

    try:
        mi = train_models(raw_df)
    except ImportError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"Model training failed: {e}")
        st.stop()

    init_state()
    inject_css()
    render_topbar()
    render_sidebar(mi, raw_df)

    st.markdown('<div class="div"></div>', unsafe_allow_html=True)

    # ── Main navigation tabs ──
    nav = st.tabs(["Predict", "📁 Batch Predict", "Analytics"])

    with nav[0]:
        inp = render_form(raw_df)
        st.markdown("<br>", unsafe_allow_html=True)
        _, mid, _ = st.columns([1, 1.4, 1])
        with mid:
            clicked = st.button("Analyse Attrition Risk", use_container_width=True)
        if clicked:
            run_prediction(mi, raw_df, inp)
        if st.session_state.result is not None:
            st.markdown('<div class="div"></div>', unsafe_allow_html=True)
            render_results()

    with nav[1]:
        render_batch(mi)

    with nav[2]:
        render_analytics(raw_df, mi)


if __name__ == "__main__":
    main()
