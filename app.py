import json
import pandas as pd
import streamlit as st
import yaml

from engine import (
    parse_sessions, make_time_index, baseline_load_curve,
    greedy_optimize_schedule, Tariff, estimate_costs
)

st.set_page_config(page_title="EV Energy OS — Depot Simulation Demo", layout="wide")

@st.cache_data
def load_default_sessions() -> pd.DataFrame:
    return pd.read_csv("data/sessions.csv")

@st.cache_data
def load_config() -> dict:
    return yaml.safe_load(open("config.yaml", "r"))

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

st.title("EV Energy OS — Depot Simulation + Optimization (V0.1)")
st.caption("Upload depot charging sessions, simulate baseline vs optimized depot load, and view a savings range (conservative → expected → aggressive).")

cfg = load_config()

with st.sidebar:
    st.header("Inputs")

    uploaded = st.file_uploader("Upload sessions.csv", type=["csv"])
    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        st.success("Uploaded sessions.csv")
    else:
        raw = load_default_sessions()
        st.info("Using bundled sample dataset (data/sessions.csv)")

    st.subheader("Depot constraints")
    time_bin = st.number_input("Time bin (minutes)", min_value=1, max_value=60, value=int(cfg["time_bin_minutes"]), step=1)
    max_concurrent = st.number_input("Max concurrent chargers", min_value=1, max_value=200, value=int(cfg["max_concurrent_chargers"]), step=1)

    st.subheader("Tariff (simple)")
    flat_rate = st.number_input("Flat energy rate (per kWh)", min_value=0.0, value=float(cfg["tariff"]["flat_energy_rate_per_kwh"]), step=0.5)
    demand_rate = st.number_input("Demand charge (per kW)", min_value=0.0, value=float(cfg["tariff"]["demand_charge_per_kw"]), step=10.0)

    st.subheader("Savings scenarios (depot power cap kW)")
    cons_cap = st.number_input("Conservative cap (kW)", min_value=1.0, value=float(cfg["savings_scenarios"]["conservative"]["depot_power_cap_kw"]), step=10.0)
    exp_cap = st.number_input("Expected cap (kW)", min_value=1.0, value=float(cfg["savings_scenarios"]["expected"]["depot_power_cap_kw"]), step=10.0)
    agg_cap = st.number_input("Aggressive cap (kW)", min_value=1.0, value=float(cfg["savings_scenarios"]["aggressive"]["depot_power_cap_kw"]), step=10.0)

    run = st.button("Run simulation", type="primary")

sessions = parse_sessions(raw)

if sessions.empty:
    st.error("No valid sessions found. Check earliest_start and latest_end columns.")
    st.stop()

t0 = sessions["earliest_start"].min().floor("D")
t1 = t0 + pd.Timedelta(hours=24)
idx = make_time_index(t0, t1, int(time_bin))

tariff = Tariff(
    flat_energy_rate_per_kwh=float(flat_rate),
    demand_charge_per_kw=float(demand_rate),
    tou_blocks=cfg["tariff"].get("tou_blocks", []),
)

colA, colB = st.columns([1.1, 1])

with colA:
    st.subheader("Dataset preview")
    st.dataframe(sessions.head(25), use_container_width=True)

with colB:
    st.subheader("Quick stats")
    st.metric("Sessions", int(len(sessions)))
    st.metric("Total energy requested (kWh)", round(float(sessions["energy_kwh"].sum()), 2))
    st.metric("Avg max kW", round(float(sessions["max_kw"].mean()), 2))

if run:
    baseline = baseline_load_curve(sessions, idx, int(time_bin))
    base_costs = estimate_costs(baseline, tariff, int(time_bin))

    def scenario(cap_kw: float):
        schedule, load = greedy_optimize_schedule(
            sessions=sessions,
            time_index=idx,
            bin_minutes=int(time_bin),
            depot_power_cap_kw=float(cap_kw),
            max_concurrent_chargers=int(max_concurrent),
        )
        costs = estimate_costs(load, tariff, int(time_bin))
        savings = {
            "peak_kw_reduction_pct": (1 - costs["peak_kw"]/base_costs["peak_kw"]) * 100 if base_costs["peak_kw"] else 0,
            "total_cost_savings": base_costs["total_cost"] - costs["total_cost"],
            "total_cost_savings_pct": (1 - costs["total_cost"]/base_costs["total_cost"]) * 100 if base_costs["total_cost"] else 0,
        }
        return schedule, load, costs, savings

    cons = scenario(cons_cap)
    exp = scenario(exp_cap)
    agg = scenario(agg_cap)

    st.divider()
    st.subheader("Load curves")
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Baseline (charge ASAP)")
        st.line_chart(baseline)
    with c2:
        st.caption("Optimized (expected scenario)")
        st.line_chart(exp[1])

    st.divider()
    st.subheader("Savings range")
    rows = []
    for name, cap, pack in [
        ("Conservative", cons_cap, cons),
        ("Expected", exp_cap, exp),
        ("Aggressive", agg_cap, agg),
    ]:
        costs = pack[2]; savings = pack[3]
        rows.append({
            "Scenario": name,
            "Depot cap (kW)": float(cap),
            "Peak kW": round(float(costs["peak_kw"]), 2),
            "Peak reduction %": round(float(savings["peak_kw_reduction_pct"]), 2),
            "Total cost": round(float(costs["total_cost"]), 2),
            "Savings": round(float(savings["total_cost_savings"]), 2),
            "Savings %": round(float(savings["total_cost_savings_pct"]), 2),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.subheader("Downloads")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button("Expected schedule CSV", data=to_csv_bytes(exp[0]), file_name="optimized_schedule_expected.csv", mime="text/csv")
    with d2:
        st.download_button("Conservative schedule CSV", data=to_csv_bytes(cons[0]), file_name="optimized_schedule_conservative.csv", mime="text/csv")
    with d3:
        st.download_button("Aggressive schedule CSV", data=to_csv_bytes(agg[0]), file_name="optimized_schedule_aggressive.csv", mime="text/csv")

    summary = {
        "baseline": base_costs,
        "scenarios": {
            "conservative": {"depot_power_cap_kw": cons_cap, "costs": cons[2], "savings": cons[3]},
            "expected": {"depot_power_cap_kw": exp_cap, "costs": exp[2], "savings": exp[3]},
            "aggressive": {"depot_power_cap_kw": agg_cap, "costs": agg[2], "savings": agg[3]},
        }
    }
    st.download_button("Summary JSON", data=json.dumps(summary, indent=2).encode("utf-8"), file_name="summary.json", mime="application/json")
else:
    st.info("Set parameters on the left and click **Run simulation**.")
