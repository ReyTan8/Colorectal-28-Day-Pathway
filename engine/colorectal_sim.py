from __future__ import annotations
from typing import Dict, List
import numpy as np
import pandas as pd


# =============================================================================
# DEFAULT PARAMETERS
# =============================================================================

def make_default_params() -> Dict:
    days_per_month = 28
    n_referrals_monthly = 400
    return {
        "seed": 42,
        "days_per_month": days_per_month,
        "n_referrals_monthly": n_referrals_monthly,

        # Routing (must sum to 1)
        "routing": {
            "OP_F2F": 0.35,
            "OP_Tel": 0.20,
            "IDA_OP": 0.25,
            "Endoscopy_Sulis": 0.20,
        },

        # OP diagnostic split AND retest split
        "p_op_to_endoscopy": 0.80,
        "p_op_to_ct": 0.20,

        # YES branching
        "p_yes_direct_stop": 0.30,
        "p_yes_histology": 0.70,

        # Daily quotas (capacity)
        "daily_quotas": {
            "OP_F2F": 8,
            "OP_Tel": 5,
            "IDA_OP": 6,
            "Endoscopy_NHS": 10,
            "Endoscopy_Sulis": 15,
            "CT_CTC": 15,
            "Histology": 10,
            "Patient_Informed": 999,
        },

        # Working days (Mon–Sun)
        "working_days": {
            "OP_F2F": [1,1,1,1,1,0,0],
            "OP_Tel": [1,1,1,1,1,0,0],
            "IDA_OP": [1,1,1,1,1,0,0],
            "Endoscopy_NHS": [1,1,1,1,1,0,0],
            "Endoscopy_Sulis": [1,1,1,1,1,0,0],
            "CT_CTC": [1,1,1,1,1,0,0],
            "Histology": [1,1,1,1,1,0,0],
            "Patient_Informed": [1,1,1,1,1,0,0],
        },

        # Scan sufficiency + IDA bundling
        "p_scan_sufficient": 0.70,
        "p_ida_same_day": 0.60,

        # Max retests
        "max_retests": 2,

        # Patient communication delay
        "comm_delay_days": 1,

        # Durations
        "multi_day_durations": {
            "Histology": 5
        },

        # Horizon
        "warm_up_period": 0,
        "sim_duration": days_per_month * 3,
        "number_of_runs": 5,
        "exclude_warmup_from_metrics": True,

        # Backlogs
        "initial_patients": 50,
        "initial_backlog_op_f2f": 0,
        "initial_backlog_op_tel": 0,
        "initial_backlog_ida_op": 0,
        "initial_backlog_endoscopy_nhs": 0,
        "initial_backlog_endoscopy_sulis": 0,
        "initial_backlog_ct": 0,
    }


# =============================================================================
# VALIDATION
# =============================================================================

def validate_params(params: Dict):
    # Routing sum
    if abs(sum(params["routing"].values()) - 1) > 1e-6:
        raise ValueError("Routing probabilities must sum to 1.")

    # OP split
    if abs(params["p_op_to_endoscopy"] + params["p_op_to_ct"] - 1) > 1e-6:
        raise ValueError("p_op_to_endoscopy + p_op_to_ct must equal 1.0")

    # YES split
    if abs(params["p_yes_direct_stop"] + params["p_yes_histology"] - 1) > 1e-6:
        raise ValueError("p_yes_direct_stop + p_yes_histology must equal 1.0")

    # Quotas
    dq = params["daily_quotas"]
    required = [
        "OP_F2F","OP_Tel","IDA_OP",
        "Endoscopy_NHS","Endoscopy_Sulis",
        "CT_CTC","Histology","Patient_Informed"
    ]
    for k in required:
        if k not in dq:
            raise ValueError(f"Missing daily quota: {k}")
        if dq[k] < 0:
            raise ValueError(f"daily_quotas[{k}] must be >= 0")

    # Probabilities
    if not 0 <= params["p_scan_sufficient"] <= 1:
        raise ValueError("p_scan_sufficient must be 0–1.")
    if not 0 <= params["p_ida_same_day"] <= 1:
        raise ValueError("p_ida_same_day must be 0–1.")

    # Other
    if params["max_retests"] < 0:
        raise ValueError("max_retests must be >= 0")
    if params["comm_delay_days"] < 0:
        raise ValueError("comm_delay_days must be >= 0")

    # Working days
    wd = params["working_days"]
    for step, week in wd.items():
        if len(week) != 7:
            raise ValueError(f"working_days[{step}] must have 7 entries.")
        if any(d not in (0,1) for d in week):
            raise ValueError(f"working_days[{step}] must contain only 0/1 values.")

    # Backlogs non-negative
    for bk in [
        "initial_patients",
        "initial_backlog_op_f2f",
        "initial_backlog_op_tel",
        "initial_backlog_ida_op",
        "initial_backlog_endoscopy_nhs",
        "initial_backlog_endoscopy_sulis",
        "initial_backlog_ct",
    ]:
        if params[bk] < 0:
            raise ValueError(f"{bk} must be >= 0")


# =============================================================================
# CONSTANTS
# =============================================================================

STEPS = [
    "OP_F2F","OP_Tel","IDA_OP",
    "Endoscopy_NHS","Endoscopy_Sulis",
    "CT_CTC","Histology",
    "Patient_Informed"
]
DECISION = "DECISION"


# =============================================================================
# HELPERS
# =============================================================================

def _assign_route(rng, routing):
    r = rng.random()
    cum = 0
    for k,v in routing.items():
        cum += v
        if r < cum:
            return k
    return list(routing.keys())[-1]


def _new_patient(pid, day, rng, params):
    return {
        "id": pid,
        "start_day": day,
        "end_day": None,
        "route": _assign_route(rng, params["routing"]),
        "ida_endo_done": False,
        "ida_ct_done": False,
        "retests": 0,
    }


def _initial_steps_for_route(route):
    return [route]


# =============================================================================
# CORE SIMULATION (ONE RUN)
# =============================================================================

def run_one_model(params: Dict, run_number: int):
    validate_params(params)
    rng = np.random.default_rng(params["seed"] + run_number)

    warm = params["warm_up_period"]
    sim_days = params["sim_duration"]
    horizon = warm + sim_days

    dq_base = params["daily_quotas"]
    multi_day = params["multi_day_durations"]
    working_days = params["working_days"]

    # Storage
    queues_today = {s: [] for s in STEPS + [DECISION]}
    queues_next  = {s: [] for s in STEPS + [DECISION]}
    in_progress  = {s: [] for s in STEPS}
    completed = []
    next_pid = 1

    # --- Backlog monitoring structures (NEW) ---
    queue_timeseries = {s: [] for s in STEPS}
    backlog_at_warmup = {s: 0 for s in STEPS}
    backlog_at_end = {s: 0 for s in STEPS}

    # Inject routed initial patients
    for _ in range(params["initial_patients"]):
        p = _new_patient(next_pid, 0, rng, params)
        next_pid += 1
        for s in _initial_steps_for_route(p["route"]):
            queues_today[s].append(p)

    # Inject explicit per-step backlogs
    for key, step in [
        ("initial_backlog_op_f2f", "OP_F2F"),
        ("initial_backlog_op_tel", "OP_Tel"),
        ("initial_backlog_ida_op", "IDA_OP"),
        ("initial_backlog_endoscopy_nhs", "Endoscopy_NHS"),
        ("initial_backlog_endoscopy_sulis", "Endoscopy_Sulis"),
        ("initial_backlog_ct", "CT_CTC"),
    ]:
        for _ in range(params[key]):
            p = _new_patient(next_pid, 0, rng, params)
            next_pid += 1
            queues_today[step].append(p)

    # Stats
    queue_days = {s: 0 for s in STEPS}
    queue_days_meas = {s: 0 for s in STEPS}
    starts_count = {s: 0 for s in STEPS}
    starts_count_meas = {s: 0 for s in STEPS}

    exclude_warmup = params.get("exclude_warmup_from_metrics", True)

    lam = 0
    if params["n_referrals_monthly"] and params["days_per_month"]:
        lam = params["n_referrals_monthly"] / params["days_per_month"]

    # -------------------------------------------------------------------------
    # DAILY LOOP
    # -------------------------------------------------------------------------
    for day in range(horizon):
        weekday = day % 7

        # move next→today
        for s in queues_today:
            queues_today[s].extend(queues_next[s])
            queues_next[s].clear()

        # tick multi-day jobs
        for step, jobs in in_progress.items():
            new_jobs = []
            for (p, rem) in jobs:
                rem -= 1
                if rem <= 0:
                    if step == "Histology":
                        queues_today["Patient_Informed"].append(p)
                    elif step == "Patient_Informed":
                        p["end_day"] = day
                        completed.append(p)
                else:
                    new_jobs.append([p, rem])
            in_progress[step] = new_jobs

        # --- Record queue length at start of day (NEW) ---
        for s in STEPS:
            queue_timeseries[s].append(
                len(queues_today[s])
                + len(queues_next[s])
                + sum(1 for _p,_rem in in_progress[s])
            )

        # queue-days
        for s in STEPS:
            queue_days[s] += len(queues_today[s])
        if exclude_warmup and day >= warm:
            for s in STEPS:
                queue_days_meas[s] += len(queues_today[s])

        # new arrivals
        if lam > 0:
            for _ in range(rng.poisson(lam)):
                p = _new_patient(next_pid, day, rng, params)
                next_pid += 1
                for s in _initial_steps_for_route(p["route"]):
                    queues_today[s].append(p)

        # capacity (zero on non-working days)
        remaining = {
            step: (dq_base[step] if working_days[step][weekday] == 1 else 0)
            for step in dq_base
        }

        # ---------------------------------------------------------------------
        # PROCESSING LOGIC (all steps, unchanged)
        # ---------------------------------------------------------------------

        # OP_F2F
        if remaining["OP_F2F"] > 0 and queues_today["OP_F2F"]:
            take = min(remaining["OP_F2F"], len(queues_today["OP_F2F"]))
            batch = queues_today["OP_F2F"][:take]
            queues_today["OP_F2F"] = queues_today["OP_F2F"][take:]
            remaining["OP_F2F"] -= take
            starts_count["OP_F2F"] += take
            if exclude_warmup and day >= warm:
                starts_count_meas["OP_F2F"] += take
            for p in batch:
                if rng.random() < params["p_op_to_endoscopy"]:
                    queues_next["Endoscopy_NHS"].append(p)
                else:
                    queues_next["CT_CTC"].append(p)

        # OP_Tel
        if remaining["OP_Tel"] > 0 and queues_today["OP_Tel"]:
            take = min(remaining["OP_Tel"], len(queues_today["OP_Tel"]))
            batch = queues_today["OP_Tel"][:take]
            queues_today["OP_Tel"] = queues_today["OP_Tel"][take:]
            remaining["OP_Tel"] -= take
            starts_count["OP_Tel"] += take
            if exclude_warmup and day >= warm:
                starts_count_meas["OP_Tel"] += take
            for p in batch:
                if rng.random() < params["p_op_to_endoscopy"]:
                    queues_next["Endoscopy_NHS"].append(p)
                else:
                    queues_next["CT_CTC"].append(p)

        # IDA_OP
        if remaining["IDA_OP"] > 0 and queues_today["IDA_OP"]:
            take = min(remaining["IDA_OP"], len(queues_today["IDA_OP"]))
            batch = queues_today["IDA_OP"][:take]
            queues_today["IDA_OP"] = queues_today["IDA_OP"][take:]
            remaining["IDA_OP"] -= take
            starts_count["IDA_OP"] += take
            if exclude_warmup and day >= warm:
                starts_count_meas["IDA_OP"] += take

            for p in batch:
                bundle = (
                    rng.random() < params["p_ida_same_day"]
                    and remaining["Endoscopy_NHS"] > 0
                    and remaining["CT_CTC"] > 0
                )
                if bundle:
                    remaining["Endoscopy_NHS"] -= 1
                    remaining["CT_CTC"] -= 1
                    p["ida_endo_done"] = True
                    p["ida_ct_done"] = True
                    queues_next[DECISION].append(p)
                else:
                    queues_next["Endoscopy_NHS"].append(p)

        # Endoscopy_Sulis
        if remaining["Endoscopy_Sulis"] > 0 and queues_today["Endoscopy_Sulis"]:
            take = min(remaining["Endoscopy_Sulis"], len(queues_today["Endoscopy_Sulis"]))
            batch = queues_today["Endoscopy_Sulis"][:take]
            queues_today["Endoscopy_Sulis"] = queues_today["Endoscopy_Sulis"][take:]
            remaining["Endoscopy_Sulis"] -= take
            starts_count["Endoscopy_Sulis"] += take
            if exclude_warmup and day >= warm:
                starts_count_meas["Endoscopy_Sulis"] += take
            for p in batch:
                queues_next[DECISION].append(p)

        # Endoscopy_NHS
        if remaining["Endoscopy_NHS"] > 0 and queues_today["Endoscopy_NHS"]:
            take = min(remaining["Endoscopy_NHS"], len(queues_today["Endoscopy_NHS"]))
            batch = queues_today["Endoscopy_NHS"][:take]
            queues_today["Endoscopy_NHS"] = queues_today["Endoscopy_NHS"][take:]
            remaining["Endoscopy_NHS"] -= take
            starts_count["Endoscopy_NHS"] += take
            if exclude_warmup and day >= warm:
                starts_count_meas["Endoscopy_NHS"] += take

            for p in batch:
                if p["route"] == "IDA_OP":
                    p["ida_endo_done"] = True
                    if not p["ida_ct_done"]:
                        queues_next["CT_CTC"].append(p)
                    else:
                        queues_next[DECISION].append(p)
                else:
                    queues_next[DECISION].append(p)

        # CT_CTC
        if remaining["CT_CTC"] > 0 and queues_today["CT_CTC"]:
            take = min(remaining["CT_CTC"], len(queues_today["CT_CTC"]))
            batch = queues_today["CT_CTC"][:take]
            queues_today["CT_CTC"] = queues_today["CT_CTC"][take:]
            remaining["CT_CTC"] -= take
            starts_count["CT_CTC"] += take
            if exclude_warmup and day >= warm:
                starts_count_meas["CT_CTC"] += take

            for p in batch:
                p["ida_ct_done"] = True
                if p["route"] == "IDA_OP":
                    if p["ida_endo_done"]:
                        queues_next[DECISION].append(p)
                    else:
                        queues_next["Endoscopy_NHS"].append(p)
                else:
                    queues_next[DECISION].append(p)

        # Histology
        if remaining["Histology"] > 0 and queues_today["Histology"]:
            take = min(remaining["Histology"], len(queues_today["Histology"]))
            batch = queues_today["Histology"][:take]
            queues_today["Histology"] = queues_today["Histology"][take:]
            remaining["Histology"] -= take
            starts_count["Histology"] += take
            if exclude_warmup and day >= warm:
                starts_count_meas["Histology"] += take

            dur = max(1, multi_day["Histology"])
            for p in batch:
                in_progress["Histology"].append([p, dur])

        # Patient_Informed
        if remaining["Patient_Informed"] > 0 and queues_today["Patient_Informed"]:
            take = min(remaining["Patient_Informed"], len(queues_today["Patient_Informed"]))
            batch = queues_today["Patient_Informed"][:take]
            queues_today["Patient_Informed"] = queues_today["Patient_Informed"][take:]
            remaining["Patient_Informed"] -= take
            starts_count["Patient_Informed"] += take
            if exclude_warmup and day >= warm:
                starts_count_meas["Patient_Informed"] += take

            dur = max(1, params["comm_delay_days"])
            for p in batch:
                in_progress["Patient_Informed"].append([p, dur])

        # DECISION router
        if queues_today[DECISION]:
            items = queues_today[DECISION][:]
            queues_today[DECISION].clear()
            for p in items:
                sufficient = (rng.random() < params["p_scan_sufficient"])
                if sufficient:
                    if rng.random() < params["p_yes_direct_stop"]:
                        queues_next["Patient_Informed"].append(p)
                    else:
                        queues_next["Histology"].append(p)
                else:
                    p["retests"] += 1
                    if p["retests"] > params["max_retests"]:
                        queues_next["Histology"].append(p)
                        continue
                    if rng.random() < params["p_op_to_endoscopy"]:
                        queues_next["Endoscopy_NHS"].append(p)
                    else:
                        queues_next["CT_CTC"].append(p)

        # snapshot at warm-up end
        if day == warm - 1:
            for s in STEPS:
                backlog_at_warmup[s] = (
                    len(queues_today[s])
                    + len(queues_next[s])
                    + sum(1 for _p,_rem in in_progress[s])
                )

    # --- final backlog snapshot ---
    for s in STEPS:
        backlog_at_end[s] = (
            len(queues_today[s])
            + len(queues_next[s])
            + sum(1 for _p,_rem in in_progress[s])
        )

    # patient-level metrics
    if exclude_warmup:
        pts = [
            p["end_day"] - p["start_day"]
            for p in completed
            if p["end_day"] is not None and p["start_day"] >= warm
        ]
    else:
        pts = [
            p["end_day"] - p["start_day"]
            for p in completed
            if p["end_day"] is not None
        ]

    mean_time = float(np.mean(pts)) if pts else 0.0
    compliance = 100 * np.mean([1 if t <= 28 else 0 for t in pts]) if pts else 0.0

    # step-level metrics
    if exclude_warmup:
        avg_wait = {
            s: (queue_days_meas[s] / starts_count_meas[s])
            if starts_count_meas[s] else 0.0
            for s in STEPS
        }
        avg_queue = {s: queue_days_meas[s] / sim_days for s in STEPS}
    else:
        avg_wait = {
            s: (queue_days[s] / starts_count[s]) if starts_count[s] else 0.0
            for s in STEPS
        }
        avg_queue = {s: queue_days[s] / horizon for s in STEPS}

    return (
        mean_time,
        compliance,
        avg_wait,
        avg_queue,
        pts,
        queue_timeseries,      # NEW
        backlog_at_warmup,     # NEW
        backlog_at_end         # NEW
    )


# =============================================================================
# MULTI-RUN WRAPPER
# =============================================================================

def run_trial(params: Dict):
    validate_params(params)

    rows = []
    waits = []
    queues = []
    all_ptimes = []

    q_ts_runs = []     # NEW
    warm_snaps = []    # NEW
    end_snaps = []     # NEW

    for run in range(params["number_of_runs"]):
        (
            mean_t,
            comp,
            avg_w,
            avg_q,
            p_times,
            q_ts,
            snap_warm,
            snap_end
        ) = run_one_model(params, run)

        rows.append({
            "Run": run,
            "Mean Pathway Time (days)": mean_t,
            "28-Day Compliance (%)": comp,
        })

        waits.append(avg_w)
        queues.append(avg_q)
        all_ptimes.extend(p_times)

        q_ts_runs.append(q_ts)
        warm_snaps.append(snap_warm)
        end_snaps.append(snap_end)

    df = pd.DataFrame(rows)
    df_wait = pd.DataFrame(waits)
    df_queue = pd.DataFrame(queues)

    # average per-step metrics
    avg_waits = df_wait.mean().to_dict()
    avg_queues = df_queue.mean().to_dict()

    # --- average snapshots (NEW) ---
    def _mean_dict(dict_list):
        if not dict_list:
            return {}
        return pd.DataFrame(dict_list).mean().to_dict()

    avg_warmup_backlog = _mean_dict(warm_snaps)
    avg_end_backlog = _mean_dict(end_snaps)

    # --- average backlog time-series (NEW) ---
    avg_q_ts = {}
    if q_ts_runs:
        steps = STEPS
        horizon = len(next(iter(q_ts_runs[0].values())))
        for s in steps:
            mat = np.array([run[s] for run in q_ts_runs])
            avg_q_ts[s] = mat.mean(axis=0).tolist()

    return (
        df,
        avg_waits,
        avg_queues,
        all_ptimes,
        avg_q_ts,             # NEW
        avg_warmup_backlog,   # NEW
        avg_end_backlog       # NEW
    )