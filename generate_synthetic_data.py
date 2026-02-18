import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_synthetic_sessions(start_dt: datetime, hours: int=24, n_sessions: int=140, seed: int=7) -> pd.DataFrame:
    random.seed(seed); np.random.seed(seed)
    rows=[]
    horizon_end = start_dt + timedelta(hours=hours)
    for i in range(n_sessions):
        arrival_hour = int(np.clip(np.random.normal(loc=19.0, scale=2.5), 0, 23))
        arrival_min = int(np.random.choice([0,5,10,15,20,25,30,35,40,45,50,55]))
        arrival = start_dt.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(hours=arrival_hour, minutes=arrival_min)
        window_hours = float(np.clip(np.random.normal(loc=6.0, scale=2.0), 2.0, 10.0))
        latest_end = min(arrival + timedelta(hours=window_hours), horizon_end)
        energy_kwh = float(np.clip(np.random.gamma(shape=3.0, scale=10.0), 10.0, 70.0))
        max_kw = float(np.random.choice([7.4, 11.0, 22.0, 30.0], p=[0.35,0.30,0.25,0.10]))
        rows.append({
            "session_id": f"S{i+1:04d}",
            "vehicle_id": f"V{random.randint(1,260):04d}",
            "earliest_start": arrival.strftime("%Y-%m-%d %H:%M:%S"),
            "latest_end": latest_end.strftime("%Y-%m-%d %H:%M:%S"),
            "energy_kwh": round(energy_kwh,2),
            "max_kw": round(max_kw,1),
        })
    return pd.DataFrame(rows).sort_values("earliest_start").reset_index(drop=True)

if __name__=="__main__":
    start = datetime(2026,2,17,0,0,0)
    df = generate_synthetic_sessions(start_dt=start, hours=24, n_sessions=140)
    df.to_csv("data/sessions.csv", index=False)
    print("Wrote data/sessions.csv")
