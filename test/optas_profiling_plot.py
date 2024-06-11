import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.stats import linregress

def compute_seconds_per_point(data_json_path):
    # unpack json
    with open(data_json_path) as f:
        data = json.load(f)

    horizons = [float(k) for k in data.keys()]
    dts = [float(k) for k in data[str(horizons[0])].keys()]

    # need to invert the json for easy plotting
    dt_times = np.zeros((len(dts), len(horizons)))
    for i in range(len(horizons)):
        for j in range(len(dts)):
            dt_times[j, i] = np.mean(data[str(horizons[i])][str(dts[j])])

    # generate number of points per trajectory
    n_points = np.zeros_like(dt_times)
    for i in range(len(horizons)):
        for j in range(len(dts)):
            n_points[j, i] = horizons[i] // dts[j]

    # compute average seconds required per point planned
    seconds_per_point = np.zeros(len(dts))
    for i in range(len(dts)):
        times = dt_times[i]
        n = n_points[i]
        seconds_per_point[i] = linregress(n, times).slope

    print(f"Mean seconds per point: {np.mean(seconds_per_point)}")


def plot_planning_time(data_json_path):
    # unpack json
    with open(data_json_path) as f:
        data = json.load(f)

    horizons = [float(k) for k in data.keys()]
    dts = [float(k) for k in data[str(horizons[0])].keys()]

    # need to invert the json for easy plotting
    dt_times = np.zeros((len(dts), len(horizons)))
    for i in range(len(horizons)):
        for j in range(len(dts)):
            dt_times[j, i] = np.mean(data[str(horizons[i])][str(dts[j])])

    # plotting
    f = plt.figure()
    a = plt.axes()

    for i in range(len(dts)):
        times = dt_times[i]
        a.plot(horizons, times)

    plt.xlabel("Time Horizon (s)")
    plt.ylabel("Planning Time (s)")
    plt.grid()

    legend_entries = [f"dt = {dt}" for dt in dts]
    plt.legend(legend_entries)

    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_json", type=str)
    parser.add_argument("--plot", action="store_true")

    args = parser.parse_args()
    compute_seconds_per_point(args.data_json)
    if args.plot:
        plot_planning_time(args.data_json)
