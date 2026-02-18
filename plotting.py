import matplotlib.pyplot as plt
import pandas as pd

def plot_load_curve(load: pd.Series, title: str, outpath: str):
    plt.figure()
    plt.plot(load.index, load.values)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("kW")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
