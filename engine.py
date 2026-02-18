from __future__ import annotations
import pandas as pd
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, List, Tuple

@dataclass
class Tariff:
    flat_energy_rate_per_kwh: float
    demand_charge_per_kw: float
    tou_blocks: List[dict]

def parse_sessions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["earliest_start"] = pd.to_datetime(out["earliest_start"], format="mixed")
    out["latest_end"] = pd.to_datetime(out["latest_end"], format="mixed")
    return out[out["latest_end"] > out["earliest_start"]].copy()

def make_time_index(start: pd.Timestamp, end: pd.Timestamp, bin_minutes: int) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq=f"{bin_minutes}min", inclusive="left")

def tou_rate_for_ts(ts: pd.Timestamp, tariff: Tariff) -> float:
    if not tariff.tou_blocks:
        return float(tariff.flat_energy_rate_per_kwh)
    h = ts.hour + ts.minute/60.0
    for b in tariff.tou_blocks:
        if b["start_hour"] <= h < b["end_hour"]:
            return float(b["rate_per_kwh"])
    return float(tariff.flat_energy_rate_per_kwh)

def baseline_load_curve(sessions: pd.DataFrame, time_index: pd.DatetimeIndex, bin_minutes: int) -> pd.Series:
    load = pd.Series(0.0, index=time_index)
    bin_hours = bin_minutes/60.0
    for _, s in sessions.iterrows():
        needed_kwh = float(s["energy_kwh"]); kw = float(s["max_kw"])
        t = s["earliest_start"]; end = s["latest_end"]
        while needed_kwh > 1e-6 and t < end:
            if t in load.index:
                deliver = min(needed_kwh, kw*bin_hours)
                load.loc[t] += deliver/bin_hours
                needed_kwh -= deliver
            t = t + timedelta(minutes=bin_minutes)
    return load

def greedy_optimize_schedule(
    sessions: pd.DataFrame,
    time_index: pd.DatetimeIndex,
    bin_minutes: int,
    depot_power_cap_kw: float,
    max_concurrent_chargers: int
) -> Tuple[pd.DataFrame, pd.Series]:
    bin_hours = bin_minutes/60.0
    load = pd.Series(0.0, index=time_index)
    chargers = pd.Series(0, index=time_index)
    rows=[]
    ss = sessions.copy()
    ss["window_bins"] = ((ss["latest_end"]-ss["earliest_start"]).dt.total_seconds()/60/bin_minutes).clip(lower=1)
    ss["tightness"] = ss["energy_kwh"]/(ss["max_kw"]*ss["window_bins"]*bin_hours)
    ss = ss.sort_values(["tightness","earliest_start"], ascending=[False,True])

    for _, s in ss.iterrows():
        bins = [t for t in time_index if (t>=s["earliest_start"] and t<s["latest_end"])]
        if not bins: 
            continue
        needed_kwh = float(s["energy_kwh"]); kw = float(s["max_kw"])
        while needed_kwh > 1e-6:
            feasible=[]
            for t in bins:
                if chargers.loc[t] >= max_concurrent_chargers: 
                    continue
                if load.loc[t] + kw > depot_power_cap_kw:
                    continue
                feasible.append((load.loc[t] + kw, load.loc[t], t))
            if not feasible:
                break
            feasible.sort(key=lambda x: (x[0], x[1]))
            t = feasible[0][2]
            deliver = min(needed_kwh, kw*bin_hours)
            used_kw = deliver/bin_hours
            load.loc[t] += used_kw
            chargers.loc[t] += 1
            rows.append({
                "session_id": s["session_id"],
                "vehicle_id": s["vehicle_id"],
                "bin_start": t,
                "kw": round(float(used_kw),3),
                "kwh": round(float(deliver),3),
            })
            needed_kwh -= deliver
            bins.remove(t)

    schedule = pd.DataFrame(rows)
    if not schedule.empty:
        schedule["bin_start"] = pd.to_datetime(schedule["bin_start"])
    return schedule, load

def estimate_costs(load_kw: pd.Series, tariff: Tariff, bin_minutes: int) -> Dict[str,float]:
    bin_hours = bin_minutes/60.0
    total_kwh=0.0; energy_cost=0.0
    for ts, kw in load_kw.items():
        kwh = float(kw)*bin_hours
        total_kwh += kwh
        energy_cost += kwh * tou_rate_for_ts(ts, tariff)
    peak_kw = float(load_kw.max()) if len(load_kw) else 0.0
    demand_cost = peak_kw * float(tariff.demand_charge_per_kw)
    return {
        "total_kwh": total_kwh,
        "peak_kw": peak_kw,
        "energy_cost": energy_cost,
        "demand_charge_cost": demand_cost,
        "total_cost": energy_cost + demand_cost,
    }
