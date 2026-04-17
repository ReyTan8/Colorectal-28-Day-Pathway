from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Optional
import streamlit as st
import pandas as pd
import numpy as np

# Optional charting
try:
    import altair as alt
    ALT_AVAILABLE = True
except Exception:
    ALT_AVAILABLE = False

# Optional graph rendering for pathway map
try:
    from graphviz import Digraph
    GV_AVAILABLE = True
except Exception:
    GV_AVAILABLE = False

# ---- IMPORT UPDATED SIMULATOR (patched colorectal_sim.py) ----
from engine.colorectal_sim import (
    make_default_params,
    run_trial,
    validate_params,
)

APP_VERSION = "v12-unified-advanced-settings+dark-compact"


# =============================================================================
# THEME / PALETTE (Dark/Light aware)
# =============================================================================

def _is_dark_theme() -> bool:
    """Return True if the current Streamlit theme base is 'dark'."""
    try:
        return (st.get_option("theme.base") or "light").lower() == "dark"
    except Exception:
        return False


@dataclass
class Palette:
    bg: str
    bg_alt: str
    grid: str
    primary: str
    secondary: str
    accent: str
    bar: str
    line: str
    box_op: str
    box_endoscopy: str
    box_results: str


def _palette() -> Palette:
    if _is_dark_theme():
        return Palette(
            bg="#0e1117", bg_alt="#1b1f2a", grid="#30363d",
            primary="#9bd3ff", secondary="#7bd389", accent="#ff9e9e",
            bar="#7bd389", line="#9bd3ff",
            box_op="#20324a",          # OP nodes
            box_endoscopy="#3a2f1a",   # Endoscopy/CT nodes
            box_results="#1e2a1e",     # Histology/Patient_Informed
        )
    return Palette(
        bg="#ffffff", bg_alt="#f5f7fa", grid="#eaeaea",
        primary="#4e79a7", secondary="#59a14f", accent="#e15759",
        bar="#59a14f", line="#4e79a7",
        box_op="#e8f1fb",
        box_endoscopy="#f7f1e1",
        box_results="#eaf7ea",
    )


PAL = _palette()


# =============================================================================
# UI / SESSION HELPERS
# =============================================================================

def _normalize_result(obj: Any) -> Optional[Dict[str, Any]]:
    base = {
        "version": APP_VERSION,
        "df": None,
        "params": {},
        "avg_waits": {},
        "avg_queues": {},
        "patient_times": [],
        "extra": {},          # backlog structures
    }

    if isinstance(obj, dict):
        out = dict(base)
        for k in base:
            if k in obj:
                out[k] = obj[k]

        # Auto-locate DataFrame if not explicitly stored
        if out["df"] is None:
            for v in obj.values():
                if isinstance(v, pd.DataFrame):
                    out["df"] = v
                    break

        return out if isinstance(out["df"], pd.DataFrame) else None

    if isinstance(obj, pd.DataFrame):
        base["df"] = obj
        return base

    return None


def get_last_result(tag: str) -> Optional[Dict[str, Any]]:
    lr = st.session_state.get("results", {}).get(tag)
    return None if lr is None else _normalize_result(lr)


def set_last_result(tag: str, df, params, avg_waits, avg_queues,
                    patient_times, extra=None):
    if "results" not in st.session_state:
        st.session_state["results"] = {}

    st.session_state["results"][tag] = {
        "version": APP_VERSION,
        "df": df,
        "params": params,
        "avg_waits": avg_waits,
        "avg_queues": avg_queues,
        "patient_times": patient_times,
        "extra": extra or {},
    }


def ensure_state():
    """Initialise parameter stores for Setup (S), Scenario A, Scenario B."""
    if "params_S" not in st.session_state:
        st.session_state["params_S"] = make_default_params()
    if "params_A" not in st.session_state:
        st.session_state["params_A"] = make_default_params()
    if "params_B" not in st.session_state:
        st.session_state["params_B"] = make_default_params()
    if "results" not in st.session_state:
        st.session_state["results"] = {}

ensure_state()


# =============================================================================
# RESOURCE TABLE
# =============================================================================

RESOURCE_ROWS = [
    ("OP_F2F", "OP F2F", "initial_backlog_op_f2f", True),
    ("OP_Tel", "OP Tel", "initial_backlog_op_tel", True),
    ("IDA_OP", "IDA OP", "initial_backlog_ida_op", True),
    ("Endoscopy_NHS", "Endoscopy NHS", "initial_backlog_endoscopy_nhs", True),
    ("Endoscopy_Sulis", "Endoscopy Sulis", "initial_backlog_endoscopy_sulis", True),
    ("CT_CTC", "CT/CTC", "initial_backlog_ct", True),
    ("Histology", "Histology", None, False),
    ("Patient_Informed", "Patient Informed", None, False),
]

DAY_NAMES = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]


def render_resource_table(prefix: str, params: Dict) -> Dict:
    dq = params["daily_quotas"]
    wd = params["working_days"]

    st.subheader("Resources: Throughput • Backlog • Working Days")

    header = st.columns([1.6, 1.1, 1.1])
    header[0].markdown("**Resource**")
    header[1].markdown("**Daily throughput**")
    header[2].markdown("**Initial backlog**")

    for idx, (step_key, label, backlog_key, has_backlog) in enumerate(RESOURCE_ROWS):
        row = st.columns([1.6, 1.1, 1.1])
        row[0].markdown(f"**{label}**")

        dq[step_key] = row[1].number_input(
            label="",
            value=int(dq[step_key]),
            min_value=0,
            key=f"{prefix}_dq_{step_key}",
        )

        if has_backlog and backlog_key:
            params[backlog_key] = row[2].number_input(
                label="",
                value=int(params.get(backlog_key, 0)),
                min_value=0,
                key=f"{prefix}_bk_{backlog_key}",
            )
        else:
            row[2].markdown("—")

        # working days
        wdrow = st.columns([1.6, 5.5])
        wdrow[0].markdown("Working days")
        sub = wdrow[1].columns(7)

        if step_key not in wd or len(wd[step_key]) != 7:
            wd[step_key] = [1,1,1,1,1,0,0]

        for i, dname in enumerate(DAY_NAMES):
            wd[step_key][i] = sub[i].checkbox(
                dname,
                value=bool(wd[step_key][i]),
                key=f"{prefix}_wd_{step_key}_{i}"
            )

        if idx < len(RESOURCE_ROWS)-1:
            st.markdown("---")

    params["daily_quotas"] = dq
    params["working_days"] = wd
    return params


# =============================================================================
# DEMAND / CAPACITY GAUGE
# =============================================================================

def _working_day_fraction(days):
    try:
        return sum(int(bool(x)) for x in days) / 7.0
    except:
        return 5/7


def _expected_retests(p_sufficient: float, max_retests: int) -> float:
    q = max(0.0, min(1.0, 1.0 - p_sufficient))
    if q == 0:
        return 0.0
    return q * (1 - q**max(1,max_retests)) / (1 - q)


def compute_expected_daily_demand(params: Dict) -> Dict[str,float]:
    r = params["routing"]
    r1 = float(r["OP_F2F"])
    r2 = float(r["OP_Tel"])
    r3 = float(r["IDA_OP"])
    r4 = float(r["Endoscopy_Sulis"])

    p_endo = params["p_op_to_endoscopy"]
    p_ct   = params["p_op_to_ct"]
    p_suff = params["p_scan_sufficient"]
    p_yes_direct = params["p_yes_direct_stop"]

    lam = 0.0
    if params.get("n_referrals_monthly", 0) and params.get("days_per_month", 0):
        lam = params["n_referrals_monthly"] / params["days_per_month"]

    E_retests = _expected_retests(p_suff, params["max_retests"])

    en_op_f2f = r1
    en_op_tel = r2
    en_ida    = r3
    en_sulis  = r4

    en_endo_nhs = (r1+r2)*p_endo + r3 + E_retests*p_endo
    en_ct       = (r1+r2)*p_ct   + r3 + E_retests*p_ct

    forced_histology_prob = (1-p_suff)**(params["max_retests"]+1)
    en_histology = p_suff*(1-p_yes_direct) + forced_histology_prob
    en_informed  = 1.0

    return {
        "OP_F2F": lam*en_op_f2f,
        "OP_Tel": lam*en_op_tel,
        "IDA_OP": lam*en_ida,
        "Endoscopy_Sulis": lam*en_sulis,
        "Endoscopy_NHS": lam*en_endo_nhs,
        "CT_CTC": lam*en_ct,
        "Histology": lam*en_histology,
        "Patient_Informed": lam*en_informed,
    }


def render_demand_capacity_gauges(params: Dict):
    st.subheader("Throughput vs Demand (estimated)")

    demand = compute_expected_daily_demand(params)
    dq = params["daily_quotas"]
    wd = params["working_days"]

    for step, label, *_ in RESOURCE_ROWS:
        cap = float(dq.get(step,0)) * _working_day_fraction(wd.get(step,[1,1,1,1,1,0,0]))
        dem = float(demand.get(step,0))
        ratio = dem/cap if cap>0 else (999 if dem>0 else 0)
        pct = min(1.0, ratio)
        color = "#2ecc71" if ratio<=0.9 else ("#f39c12" if ratio<=1.1 else "#e76f51")

        c1,c2,c3 = st.columns([1.6,3.5,1.6])
        c1.markdown(f"**{label}**")
        c1.markdown(f"Demand: {dem:.2f}/day")

        try:
            data = pd.DataFrame({"util":[min(1.5,ratio)]})
            bar = (
                alt.Chart(data)
                .mark_bar(height=22)
                .encode(x=alt.X("util:Q", scale=alt.Scale(domain=[0,1])),
                        color=alt.value(color))
                .properties(width=300)
            )
            base = (
                alt.Chart(pd.DataFrame({"one":[1]}))
                .mark_bar(height=22, color=("#3a3f47" if _is_dark_theme() else "#eee"))
                .encode(x=alt.X("one:Q", scale=alt.Scale(domain=[0,1])))
                .properties(width=300)
            )
            c2.altair_chart(base + bar, use_container_width=False)
        except:
            c2.progress(float(pct))

        c3.markdown(
            f"Utilisation: {ratio*100:.0f}%<br>"
            f"Capacity: {cap:.1f}/day",
            unsafe_allow_html=True
        )


# =============================================================================
# PARAMETER UI (unified)
# =============================================================================

def render_params_ui(prefix: str, base_params: Optional[Dict] = None) -> Dict:
    """
    Unified UI:
    - BASIC: routing, resources, compliance target, warm-up exclusion, capacity gauge
    - ADVANCED: all 11 settings grouped together at bottom toggle
    """
    params = deepcopy(base_params) if base_params else make_default_params()

    # =========================
    # BASIC SECTION
    # =========================

    st.subheader("Referral Routing (self‑balancing)")
    r = params["routing"].copy()
    c1,c2,c3 = st.columns(3)
    r1 = c1.slider("OP F2F", 0.0,1.0, float(r["OP_F2F"]), 0.01, key=f"{prefix}_f2f")
    r2 = c2.slider("OP Tel", 0.0,1.0, float(r["OP_Tel"]), 0.01, key=f"{prefix}_tel")
    r3 = c3.slider("IDA OP", 0.0,1.0, float(r["IDA_OP"]), 0.01, key=f"{prefix}_ida")
    residual = 1.0 - (r1+r2+r3)
    if residual < 0 or residual > 1:
        st.error(f"Routing exceeds 1.0 by {abs(residual):.2f}. Adjust sliders.")
    residual = max(0, min(1, residual))
    st.info(f"Endoscopy Sulis (auto): **{residual:.2f}**")

    params["routing"] = {
        "OP_F2F": float(r1),
        "OP_Tel": float(r2),
        "IDA_OP": float(r3),
        "Endoscopy_Sulis": float(residual),
    }

    # Resource table
    params = render_resource_table(prefix, params)

    # Compliance + warm‑up exclusion in basic
    st.subheader("Performance Targets")
    d1,d2 = st.columns(2)
    params["target_compliance"] = d1.number_input(
        "Target 28‑Day Compliance (%)",
        value=float(params.get("target_compliance",70)),
        min_value=0.0, max_value=100.0, step=1.0,
        key=f"{prefix}_target"
    )
    params["exclude_warmup_from_metrics"] = d2.toggle(
        "Exclude warm‑up from metrics",
        value=bool(params.get("exclude_warmup_from_metrics", True)),
        key=f"{prefix}_exclude"
    )

    # Demand/capacity gauge
    render_demand_capacity_gauges(params)

    # =========================
    # ADVANCED SETTINGS TOGGLE
    # =========================
    st.markdown("---")
    adv_open = st.toggle(
        "Show Advanced Settings",
        value=False,
        key=f"{prefix}_adv_toggle"
    )

    if adv_open:
        st.subheader("Advanced Settings")

        # Global arrivals / seed
        g1,g2,g3 = st.columns(3)
        params["seed"] = g1.number_input(
            "Random seed", value=int(params["seed"]), min_value=0, key=f"{prefix}_seed"
        )
        params["days_per_month"] = g2.number_input(
            "Days per month", value=int(params["days_per_month"]), min_value=1, key=f"{prefix}_dpm"
        )
        params["n_referrals_monthly"] = g3.number_input(
            "Referrals per month", value=int(params["n_referrals_monthly"]),
            min_value=0, step=10, key=f"{prefix}_refpm"
        )

        # OP Diagnostic Split + Scan sufficient
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("**OP Diagnostic Split**")
            params["p_op_to_endoscopy"] = st.slider(
                "→ Endoscopy (fraction)", 0.0,1.0,
                float(params["p_op_to_endoscopy"]), 0.01,
                key=f"{prefix}_pendo"
            )
            params["p_op_to_ct"] = 1.0 - params["p_op_to_endoscopy"]
            st.caption(f"→ CT/CTC automatically: **{params['p_op_to_ct']:.2f}**")

        with c2:
            st.markdown("**Probability: scan sufficient**")
            params["p_yes_direct_stop"] = st.slider(
                "Scan sufficient → Direct communication (fraction)",
                0.0,1.0, float(params["p_yes_direct_stop"]), 0.01,
                key=f"{prefix}_psuff_direct"
            )
            params["p_yes_histology"] = 1.0 - params["p_yes_direct_stop"]
            st.caption(f"No → Histology: **{params['p_yes_histology']:.2f}**")

        # Scan interpretation and retests (all here)
        pr1,pr2,pr3 = st.columns(3)
        params["p_scan_sufficient"] = pr1.slider(
            "P(scan sufficient)", 0.0,1.0,
            float(params["p_scan_sufficient"]), 0.01,
            key=f"{prefix}_pscan"
        )
        params["p_ida_same_day"] = pr2.slider(
            "IDA same‑day bundling probability",
            0.0,1.0, float(params["p_ida_same_day"]),0.01,
            key=f"{prefix}_pida"
        )
        params["max_retests"] = pr3.number_input(
            "Max retests", value=int(params["max_retests"]), min_value=0,
            key=f"{prefix}_maxre"
        )

        # Durations + comm delay (moved here)
        d1,d2 = st.columns(2)
        mdd = params["multi_day_durations"]
        mdd["Histology"] = d1.number_input(
            "Histology duration (days)",
            value=int(mdd["Histology"]), min_value=1,
            key=f"{prefix}_histo"
        )
        params["multi_day_durations"] = mdd
        params["comm_delay_days"] = d2.number_input(
            "Communication delay (days)",
            value=int(params["comm_delay_days"]), min_value=0,
            key=f"{prefix}_comm"
        )

        # Other initialisation (moved here)
        params["initial_patients"] = st.number_input(
            "Initial routed referrals backlog (distributed by routing)",
            value=int(params["initial_patients"]), min_value=0, step=5,
            key=f"{prefix}_initp"
        )

        # Experiment horizon
        st.markdown("**Experiment Horizon**")
        h1,h2,h3 = st.columns(3)
        params["warm_up_period"] = h1.number_input(
            "Warm‑up (days)", value=int(params["warm_up_period"]), min_value=0,
            key=f"{prefix}_warm"
        )
        params["sim_duration"] = h2.number_input(
            "Measurement duration (days)",
            value=int(params["sim_duration"]), min_value=1,
            key=f"{prefix}_simd"
        )
        params["number_of_runs"] = h3.number_input(
            "Replications", value=int(params["number_of_runs"]), min_value=1,
            key=f"{prefix}_runs"
        )

    return params


# =============================================================================
# PATHWAY MAP
# =============================================================================

def _altair_conf():
    if not ALT_AVAILABLE:
        return
    try:
        alt.themes.enable("none")
    except:
        pass
    # Global aesthetic tweaks for dark/light
    base_color = "#c9d1d9" if _is_dark_theme() else "#111111"
    grid_color = PAL.grid
    alt.themes.register('m365_theme', lambda: {
        "config": {
            "view": {"strokeOpacity": 0, "continuousHeight": 300},
            "axis": {
                "labelColor": base_color,
                "titleColor": base_color,
                "gridColor": grid_color,
                "domainColor": grid_color
            },
            "legend": {"labelColor": base_color, "titleColor": base_color},
            "title": {"color": base_color},
        }
    })
    try:
        alt.themes.enable('m365_theme')
    except:
        pass


def _render_pathway_map(avg_waits: Dict[str,float]):
    st.markdown("### Pathway Overview")

    def lw(step):
        v = avg_waits.get(step, 0.0)
        return f"{step}\nAvg wait: {v:.1f} d"

    if GV_AVAILABLE:
        dot = Digraph(graph_attr={"splines":"spline","rankdir":"LR"})

        dot.node("OP_F2F", lw("OP_F2F"), shape="box", style="rounded,filled", fillcolor=PAL.box_op)
        dot.node("OP_Tel", lw("OP_Tel"), shape="box", style="rounded,filled", fillcolor=PAL.box_op)
        dot.node("IDA_OP", lw("IDA_OP"), shape="box", style="rounded,filled", fillcolor=PAL.box_op)

        dot.node("Endoscopy_Sulis", lw("Endoscopy_Sulis"), shape="box", style="rounded,filled", fillcolor=PAL.box_endoscopy)
        dot.node("Endoscopy_NHS", lw("Endoscopy_NHS"), shape="box", style="rounded,filled", fillcolor=PAL.box_endoscopy)
        dot.node("CT_CTC", lw("CT_CTC"), shape="box", style="rounded,filled", fillcolor=PAL.box_endoscopy)

        dot.node("DECISION", "DECISION", shape="diamond", style="filled",
                 fillcolor=("#2b2f36" if _is_dark_theme() else "#f0f0f0"))

        dot.node("Histology", lw("Histology"), shape="box", style="rounded,filled", fillcolor=PAL.box_results)
        dot.node("Patient_Informed", lw("Patient_Informed"), shape="box", style="rounded,filled", fillcolor=PAL.box_results)

        # edges
        dot.edge("OP_F2F","Endoscopy_NHS")
        dot.edge("OP_F2F","CT_CTC")
        dot.edge("OP_Tel","Endoscopy_NHS")
        dot.edge("OP_Tel","CT_CTC")
        dot.edge("IDA_OP","Endoscopy_NHS")

        dot.edge("Endoscopy_NHS","CT_CTC", label="IDA if CT pending", fontsize="10")
        dot.edge("Endoscopy_Sulis","DECISION")
        dot.edge("Endoscopy_NHS","DECISION")
        dot.edge("CT_CTC","DECISION")

        dot.edge("DECISION","Patient_Informed", label="YES → direct", fontsize="10")
        dot.edge("DECISION","Histology", label="YES → histology", fontsize="10")
        dot.edge("Histology","Patient_Informed")

        st.graphviz_chart(dot, use_container_width=True)
    else:
        st.info("Graphviz unavailable; skipping map.")


# =============================================================================
# RESULTS BLOCK (with compact layout tabs)
# =============================================================================

def render_results_block(tag: str, lr: Dict[str, Any], show_download: bool=True):
    df = lr["df"]
    params = lr["params"]
    avg_waits = lr["avg_waits"]
    avg_queues = lr["avg_queues"]
    patient_times = lr["patient_times"]
    extra = lr.get("extra", {})

    avg_q_ts = extra.get("avg_q_ts", {})
    snap_warm = extra.get("snap_warm", {})
    snap_end  = extra.get("snap_end", {})

    _altair_conf()

    st.markdown("### Results")

    mean_path = df["Mean Pathway Time (days)"].mean()
    mean_comp = df["28-Day Compliance (%)"].mean()
    p95 = np.percentile(patient_times,95) if patient_times else 0
    median = np.percentile(patient_times,50) if patient_times else 0
    target = params.get("target_compliance", 70)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Mean Pathway (days)", f"{mean_path:.2f}")
    if mean_comp >= target:
        c2.metric("28‑Day Compliance", f"{mean_comp:.2f}%", delta="✔ Target met")
    else:
        c2.metric("28‑Day Compliance", f"{mean_comp:.2f}%", delta="✘ Below target")
    c3.metric("Median (days)", f"{median:.1f}")
    c4.metric("95th percentile", f"{p95:.1f}")

    _render_pathway_map(avg_waits)

    # Compact layout switch for this block (defaults to global or dark theme)
    compact_default = st.session_state.get("compact_layout_global", _is_dark_theme())
    st.session_state["compact_layout"] = st.toggle(
        "Compact layout (auto)", value=st.session_state.get("compact_layout", compact_default),
        key=f"compact_layout_{tag}",
        help="Groups long sections into tabs; ON by default in Dark Mode."
    )

    if st.session_state["compact_layout"]:
        tabPR, tabHIST, tabWAIT, tabRUN, tabBACK = st.tabs(
            ["Per‑run", "Times Dist.", "Wait/Step", "By Run", "Backlog"]
        )

        with tabPR:
            st.markdown("#### Per‑run Results")
            st.dataframe(df, use_container_width=True)

        with tabHIST:
            st.markdown("#### Distribution of Patient Pathway Times")
            pt_df = pd.DataFrame({"Pathway Time (days)": patient_times})
            if ALT_AVAILABLE:
                chart = (
                    alt.Chart(pt_df)
                    .mark_bar(opacity=0.85, color=PAL.primary)
                    .encode(
                        alt.X("Pathway Time (days):Q", bin=alt.Bin(maxbins=40)),
                        alt.Y("count():Q")
                    )
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.bar_chart(pt_df)

        with tabWAIT:
            st.markdown("#### Average Wait per Step")
            wdf = pd.DataFrame({"Step": list(avg_waits.keys()), "Avg Wait (days)": list(avg_waits.values())})
            if ALT_AVAILABLE:
                chart = (
                    alt.Chart(wdf)
                    .mark_bar(color=PAL.bar)
                    .encode(x="Step:N", y="Avg Wait (days):Q")
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.bar_chart(wdf.set_index("Step"))

        with tabRUN:
            st.markdown("#### Mean Pathway Time by Run")
            if ALT_AVAILABLE:
                chart = (
                    alt.Chart(df)
                    .mark_line(point=True, color=PAL.accent)
                    .encode(x="Run:O", y="Mean Pathway Time (days):Q")
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.line_chart(df.set_index("Run")["Mean Pathway Time (days)"])

        with tabBACK:
            st.markdown("#### Backlog at Key Points")
            if snap_end:
                steps = list(snap_end.keys())
                snap_df = pd.DataFrame({
                    "Step": steps,
                    "Backlog at Warm-up Start": [snap_warm.get(s,0) for s in steps],
                    "Backlog at End of Simulation": [snap_end.get(s,0) for s in steps],
                })
                st.dataframe(snap_df, use_container_width=True)
            else:
                st.info("No backlog snapshot data.")

            st.markdown("#### Backlog Reduction Over Time")
            if avg_q_ts:
                rows = []
                for s, series in avg_q_ts.items():
                    for day,val in enumerate(series):
                        rows.append({"Day":day, "Step":s, "Backlog":val})
                ts_df = pd.DataFrame(rows)
                if ALT_AVAILABLE:
                    chart = (
                        alt.Chart(ts_df)
                        .mark_line(color=PAL.line)
                        .encode(
                            x="Day:Q", y="Backlog:Q", color="Step:N",
                            tooltip=["Day","Step","Backlog"]
                        )
                    )
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.line_chart(ts_df.pivot(index="Day", columns="Step", values="Backlog"))
            else:
                st.info("No backlog timeseries available.")
    else:
        # Stacked layout as before
        st.markdown("#### Per‑run Results")
        st.dataframe(df, use_container_width=True)

        st.markdown("#### Distribution of Patient Pathway Times")
        pt_df = pd.DataFrame({"Pathway Time (days)": patient_times})
        if ALT_AVAILABLE:
            chart = (
                alt.Chart(pt_df)
                .mark_bar(opacity=0.85, color=PAL.primary)
                .encode(
                    alt.X("Pathway Time (days):Q", bin=alt.Bin(maxbins=40)),
                    alt.Y("count():Q")
                )
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.bar_chart(pt_df)

        st.markdown("#### Average Wait per Step")
        wdf = pd.DataFrame({"Step": list(avg_waits.keys()), "Avg Wait (days)": list(avg_waits.values())})
        if ALT_AVAILABLE:
            chart = (
                alt.Chart(wdf)
                .mark_bar(color=PAL.bar)
                .encode(x="Step:N", y="Avg Wait (days):Q")
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.bar_chart(wdf.set_index("Step"))

        st.markdown("#### Mean Pathway Time by Run")
        if ALT_AVAILABLE:
            chart = (
                alt.Chart(df)
                .mark_line(point=True, color=PAL.accent)
                .encode(x="Run:O", y="Mean Pathway Time (days):Q")
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.line_chart(df.set_index("Run")["Mean Pathway Time (days)"])

        st.markdown("### Backlog at Key Points")
        if snap_end:
            steps = list(snap_end.keys())
            snap_df = pd.DataFrame({
                "Step": steps,
                "Backlog at Warm-up Start": [snap_warm.get(s,0) for s in steps],
                "Backlog at End of Simulation": [snap_end.get(s,0) for s in steps],
            })
            st.dataframe(snap_df, use_container_width=True)
        else:
            st.info("No backlog snapshot data.")

        st.markdown("### Backlog Reduction Over Time")
        if avg_q_ts:
            rows = []
            for s, series in avg_q_ts.items():
                for day,val in enumerate(series):
                    rows.append({"Day":day, "Step":s, "Backlog":val})
            ts_df = pd.DataFrame(rows)
            if ALT_AVAILABLE:
                chart = (
                    alt.Chart(ts_df)
                    .mark_line(color=PAL.line)
                    .encode(
                        x="Day:Q", y="Backlog:Q", color="Step:N",
                        tooltip=["Day","Step","Backlog"]
                    )
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.line_chart(ts_df.pivot(index="Day", columns="Step", values="Backlog"))
        else:
            st.info("No backlog timeseries available.")

    if show_download:
        st.markdown("#### Download CSV")
        st.download_button(
            "Download",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"results_{tag}.csv",
            mime="text/csv"
        )


# =============================================================================
# COMPARISON
# =============================================================================

def summary_metrics(lr: Dict[str,Any]) -> Dict[str,float]:
    df = lr["df"]
    return {
        "Mean Pathway Time (days)": float(df["Mean Pathway Time (days)"].mean()),
        "28-Day Compliance (%)": float(df["28-Day Compliance (%)"].mean()),
        "Replications": len(df),
    }


def render_side_by_side(tagL: str, lrL: Dict, tagR: str, lrR: Dict):
    colA, colB = st.columns(2)
    with colA:
        st.subheader(f"Scenario {tagL}")
        render_results_block(tagL, lrL, show_download=False)
    with colB:
        st.subheader(f"Scenario {tagR}")
        render_results_block(tagR, lrR, show_download=False)

    st.markdown("### Summary Comparison")
    a = summary_metrics(lrL)
    b = summary_metrics(lrR)
    delta = {
        "Metric": list(a.keys()),
        f"{tagL}": list(a.values()),
        f"{tagR}": [b[k] for k in a.keys()],
        "Δ (B−A)": [b[k] - a[k] for k in a.keys()],
    }
    st.dataframe(pd.DataFrame(delta), use_container_width=True)


# =============================================================================
# PAGE LAYOUT
# =============================================================================

st.set_page_config(
    page_title="Colorectal Pathway Simulator",
    page_icon="💻",
    layout="wide"
)

st.title("💻 Colorectal Pathway — Daily‑Step Simulation")

# Global compact toggle (auto ON in dark theme)
st.session_state["compact_layout_global"] = st.toggle(
    "Compact layout (auto)",
    value=st.session_state.get("compact_layout_global", _is_dark_theme()),
    help="Reduces scrolling by grouping long sections into tabs. Defaults to ON in Dark Mode."
)

st.caption(
    "**Setup → Results → Comparison.** Includes throughput/backlog, retests, delays, "
    "and patient‑level outcomes. 28‑day clock stops at Patient_Informed."
)

tab_setup, tab_results, tab_compare = st.tabs(["⚙️ Setup", "📊 Results", "🅰️🅱️ Comparison"])


# =============================================================================
# SETUP TAB
# =============================================================================

with tab_setup:
    st.header("Simulation Setup")

    sticky = st.empty()

    params_S = render_params_ui("S", base_params=st.session_state["params_S"])
    st.session_state["params_S"] = deepcopy(params_S)

    with sticky.container():
        st.markdown("### Actions")
        c1, c2 = st.columns([1,5])
        if c1.button("▶️ Run Simulation", key="run_S"):
            try:
                validate_params(st.session_state["params_S"])
                with st.spinner("Running scenario S..."):
                    df, aw, aq, pt, avg_q_ts, snap_warm, snap_end = run_trial(
                        st.session_state["params_S"]
                    )
                set_last_result(
                    "S", df, deepcopy(st.session_state["params_S"]),
                    aw, aq, pt,
                    extra={
                        "avg_q_ts": avg_q_ts,
                        "snap_warm": snap_warm,
                        "snap_end": snap_end
                    }
                )
                st.success("Scenario S complete. See **Results** tab.")
            except Exception as e:
                st.error("Simulation failed.")
                st.exception(e)


# =============================================================================
# RESULTS TAB
# =============================================================================

with tab_results:
    st.header("Results")
    lr_S = get_last_result("S")
    if lr_S and lr_S["df"] is not None and not lr_S["df"].empty:
        # Sync global compact preference to Results block instance
        st.session_state["compact_layout"] = st.session_state.get("compact_layout_global", _is_dark_theme())
        render_results_block("S", lr_S)
    else:
        st.info("Run a scenario first from the Setup tab.")


# =============================================================================
# COMPARISON TAB
# =============================================================================

with tab_compare:
    st.header("Scenario Comparison (A vs B)")

    colA, colB = st.columns(2)

    # Parameter panels
    with colA:
        st.subheader("Scenario A — Parameters")
        copyA1, copyA2 = st.columns([1,1])
        if copyA1.button("Copy Setup → A"):
            st.session_state["params_A"] = deepcopy(st.session_state["params_S"])
            st.success("Copied Setup into A.")
            st.rerun()
        if copyA2.button("Copy B → A"):
            st.session_state["params_A"] = deepcopy(st.session_state["params_B"])
            st.success("Copied B into A.")
            st.rerun()

        params_A = render_params_ui("A", base_params=st.session_state["params_A"])
        st.session_state["params_A"] = deepcopy(params_A)

    with colB:
        st.subheader("Scenario B — Parameters")
        copyB1, copyB2 = st.columns([1,1])
        if copyB1.button("Copy Setup → B"):
            st.session_state["params_B"] = deepcopy(st.session_state["params_S"])
            st.success("Copied Setup into B.")
            st.rerun()
        if copyB2.button("Copy A → B"):
            st.session_state["params_B"] = deepcopy(st.session_state["params_A"])
            st.success("Copied A into B.")
            st.rerun()

        params_B = render_params_ui("B", base_params=st.session_state["params_B"])
        st.session_state["params_B"] = deepcopy(params_B)

    st.markdown("### Run Scenarios")

    rA, rB, rBoth = st.columns([1,1,3])
    doA = rA.button("▶️ Run A")
    doB = rB.button("▶️ Run B")
    doBoth = rBoth.button("▶️▶️ Run A & B")

    try:
        if doA:
            validate_params(st.session_state["params_A"])
            with st.spinner("Running Scenario A..."):
                df, aw, aq, pt, avg_q_ts, snap_warm, snap_end = run_trial(
                    st.session_state["params_A"]
                )
            set_last_result(
                "A", df, deepcopy(st.session_state["params_A"]), aw, aq, pt,
                extra={
                    "avg_q_ts": avg_q_ts,
                    "snap_warm": snap_warm,
                    "snap_end": snap_end
                }
            )
            st.success("Scenario A completed.")

        if doB:
            validate_params(st.session_state["params_B"])
            with st.spinner("Running Scenario B..."):
                df, aw, aq, pt, avg_q_ts, snap_warm, snap_end = run_trial(
                    st.session_state["params_B"]
                )
            set_last_result(
                "B", df, deepcopy(st.session_state["params_B"]), aw, aq, pt,
                extra={
                    "avg_q_ts": avg_q_ts,
                    "snap_warm": snap_warm,
                    "snap_end": snap_end
                }
            )
            st.success("Scenario B completed.")

        if doBoth:
            # Run A
            validate_params(st.session_state["params_A"])
            with st.spinner("Running Scenario A..."):
                df, aw, aq, pt, avg_q_ts, snap_warm, snap_end = run_trial(
                    st.session_state["params_A"]
                )
            set_last_result(
                "A", df, deepcopy(st.session_state["params_A"]), aw, aq, pt,
                extra={
                    "avg_q_ts": avg_q_ts,
                    "snap_warm": snap_warm,
                    "snap_end": snap_end
                }
            )
            # Run B
            validate_params(st.session_state["params_B"])
            with st.spinner("Running Scenario B..."):
                df, aw, aq, pt, avg_q_ts, snap_warm, snap_end = run_trial(
                    st.session_state["params_B"]
                )
            set_last_result(
                "B", df, deepcopy(st.session_state["params_B"]), aw, aq, pt,
                extra={
                    "avg_q_ts": avg_q_ts,
                    "snap_warm": snap_warm,
                    "snap_end": snap_end
                }
            )
            st.success("Both scenarios completed.")

    except Exception as e:
        st.error("Simulation failed.")
        st.exception(e)

    # Display side-by-side comparison
    lrA = get_last_result("A")
    lrB = get_last_result("B")
    if lrA and lrB and not lrA["df"].empty and not lrB["df"].empty:
        # Use global compact preference also in Comparison view
        st.session_state["compact_layout"] = st.session_state.get("compact_layout_global", _is_dark_theme())
        render_side_by_side("A", lrA, "B", lrB)
    else:
        st.info("Run both scenarios to compare.")