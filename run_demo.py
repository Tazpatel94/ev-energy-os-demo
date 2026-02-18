import json, os
import pandas as pd
import yaml
from datetime import datetime
from engine import parse_sessions, make_time_index, baseline_load_curve, greedy_optimize_schedule, Tariff, estimate_costs
from plotting import plot_load_curve

def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    cfg = yaml.safe_load(open("config.yaml","r"))

    sessions_path = "data/sessions.csv"
    if not os.path.exists(sessions_path):
        from generate_synthetic_data import generate_synthetic_sessions
        start = datetime(2026,2,17,0,0,0)
        df = generate_synthetic_sessions(start_dt=start, hours=24, n_sessions=140)
        df.to_csv(sessions_path, index=False)

    sessions = parse_sessions(pd.read_csv(sessions_path))
    t0 = sessions["earliest_start"].min().floor("D")
    t1 = t0 + pd.Timedelta(hours=24)
    idx = make_time_index(t0, t1, cfg["time_bin_minutes"])

    baseline = baseline_load_curve(sessions, idx, cfg["time_bin_minutes"])
    plot_load_curve(baseline, "Baseline load curve (start charging ASAP)", "outputs/baseline_load.png")

    tariff = Tariff(**cfg["tariff"])
    summary = {"scenarios": {}}
    base_costs = estimate_costs(baseline, tariff, cfg["time_bin_minutes"])
    summary["baseline"] = base_costs

    for name, overrides in cfg["savings_scenarios"].items():
        cap = float(overrides.get("depot_power_cap_kw", cfg["depot_power_cap_kw"]))
        schedule, load = greedy_optimize_schedule(sessions, idx, cfg["time_bin_minutes"], cap, int(cfg["max_concurrent_chargers"]))
        costs = estimate_costs(load, tariff, cfg["time_bin_minutes"])
        summary["scenarios"][name] = {
            "depot_power_cap_kw": cap,
            "costs": costs,
            "savings": {
                "peak_kw_reduction_pct": (1 - costs["peak_kw"]/base_costs["peak_kw"]) * 100 if base_costs["peak_kw"] else 0,
                "total_cost_savings": base_costs["total_cost"] - costs["total_cost"],
                "total_cost_savings_pct": (1 - costs["total_cost"]/base_costs["total_cost"]) * 100 if base_costs["total_cost"] else 0,
            }
        }
        plot_load_curve(load, f"Optimized load curve ({name})", f"outputs/optimized_load_{name}.png")
        schedule.to_csv(f"outputs/optimized_schedule_{name}.csv", index=False)

    json.dump(summary, open("outputs/summary.json","w"), indent=2)
    print("Done. See outputs/")

if __name__=="__main__":
    main()
