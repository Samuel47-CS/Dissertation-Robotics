"""Episode clustering utilities for Bi-SO101 dataset analysis.

This module loads episode data from CSV files, filters joint state trajectories,
identifies grasp phases, computes a phase-weighted DTW distance matrix, and
clusters episodes using agglomerative clustering.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import medfilt, savgol_filter
from sklearn.cluster import AgglomerativeClustering
from tslearn.metrics import dtw

DATAPATH = Path("datasets/csv")
CSV_STRING = "ep{i}.csv"

"""These hyperparameters were tuned relative to the dataset used for my research.
If you are using a different dataset then use the tuning loop in data_analysis.ipynb 
over your dataset!
"""
CLUSTERS = 3

PHASE_WEIGHTS = {
    "pre_grasp": 0.4722948402020483,
    "grasp": 0.10091806117592445,
    "post_grasp": 4.538480240384448,
}

JOINT_WEIGHTS = {
    "joint_1": 0.7698794230565094,
    "joint_2": 1.2643617891099344,
    "joint_3": 0.8828660596020026,
    "joint_4": 0.8074812980086434,
    "joint_5": 2.4099800836687684,
    "joint_6": 0.7151637734507728,
    "joint_7": 0.7698794230565094,
    "joint_8": 1.2643617891099344,
    "joint_9": 0.8828660596020026,
    "joint_10": 0.8074812980086434,
    "joint_11": 2.4099800836687684,
    "joint_12": 0.7151637734507728,
}


def create_df(datapath=DATAPATH, csv_string=CSV_STRING):
    """Load episode CSV files and assemble a DataFrame."""
    datapath = Path(datapath)
    csv_files = sorted(datapath.glob("*.csv"))

    rows = []
    for episode_id, csv_file in enumerate(csv_files):
        frame = pd.read_csv(csv_file)
        states = frame.drop(columns=["timestamp"])
        states = pd.DataFrame.from_dict(median_filter_episode(states))
        states = pd.DataFrame.from_dict(savgol_episode(states))
        timestamps = frame["timestamp"]

        phases = detect_grasp_phase(states, timestamps)
        if phases is None:
            print("FAILED TO SPLIT PHASES!!! EPISODE", episode_id)
            continue

        rows.append({
            "episode_id": episode_id,
            "n_steps": frame.shape[0],
            "duration": float(timestamps.iloc[-1] - timestamps.iloc[0]),
            "states": states,
            "timestamps": timestamps,
            "phase": phases_to_array(phases, len(timestamps)),
            "valid": True,
        })

    return pd.DataFrame(rows)


def detect_grasp_phase(
    states,
    timestamps,
    joint_indices=("joint_6", "joint_12"),
    zero_thresh=3.4,
    motion_thresh=5.0,
    min_duration_s=1,
):
    """Detect pre-grasp, grasp, and post-grasp phases from joint state trajectories."""
    j1, j2 = joint_indices
    t = states.shape[0]
    mask = (
        (np.abs(states[j1]) <= zero_thresh)
        & (np.abs(states[j2]) <= zero_thresh)
    )

    segments = []
    in_segment = False
    for idx, value in enumerate(mask):
        if value and not in_segment:
            segment_start = idx
            in_segment = True
        elif not value and in_segment:
            segments.append((segment_start, idx - 1))
            in_segment = False
    if in_segment:
        segments.append((segment_start, t - 1))

    if not segments:
        return None

    valid_segments = [
        (s, e, timestamps.iloc[e] - timestamps.iloc[s])
        for s, e in segments
        if (timestamps.iloc[e] - timestamps.iloc[s]) >= min_duration_s
    ]

    if not valid_segments:
        valid_segments = [
            (s, e, timestamps.iloc[e] - timestamps.iloc[s]) for s, e in segments
        ]

    s, e, _ = max(valid_segments, key=lambda entry: entry[2])

    def has_motion(start, end):
        if start >= end:
            return False
        motion_mask = (
            (np.abs(states[j1][start:end]) >= motion_thresh)
            | (np.abs(states[j2][start:end]) >= motion_thresh)
        )
        return np.any(motion_mask)

    if not (has_motion(0, s) and has_motion(e + 1, t)):
        for s2, e2, _ in sorted(valid_segments, key=lambda entry: -entry[2]):
            if has_motion(0, s2) and has_motion(e2 + 1, t):
                s, e = s2, e2
                break

    return {
        "pre_grasp": (0, max(0, int(s) - 1)),
        "grasp": (int(s), int(e)),
        "post_grasp": (min(t - 1, int(e) + 1), t - 1),
    }


def phases_to_array(phase, num_timesteps):
    """Convert phase boundaries into a timestep-aligned phase array."""
    phase_arr = np.empty(num_timesteps, dtype=object)
    for name, (start, end) in phase.items():
        phase_arr[start : end + 1] = name
    return phase_arr


def median_filter_episode(states, kernel_size=5):
    return {
        joint: medfilt(series, kernel_size=kernel_size)
        for joint, series in states.items()
    }


def savgol_episode(states, window=9, poly=2):
    return {
        joint: savgol_filter(series, window_length=window, polyorder=poly)
        for joint, series in states.items()
    }


def trim_outliers(x, z=3.0):
    mu, sigma = np.mean(x), np.std(x)
    return np.clip(x, mu - z * sigma, mu + z * sigma)


def remove_bad_episodes(episodes_df, bad_ep_ids):
    episodes_df["valid"] = True
    episodes_df.loc[episodes_df["episode_id"].isin(bad_ep_ids), "valid"] = False
    print(f"Removed episodes {bad_ep_ids}")
    return episodes_df


def weighted_episode_dtw_distance(
    episodes_df,
    phase_weights=PHASE_WEIGHTS,
    joint_weights=JOINT_WEIGHTS,
    min_len=5,
):
    """Compute a symmetric distance matrix using weighted DTW."""
    N = len(episodes_df)
    D = np.zeros((N, N), dtype=float)
    all_phases = sorted({phase for row in episodes_df["phase"] for phase in row})
    joint_names = sorted(episodes_df.iloc[0]["states"].keys())

    for i in range(N):
        row_i = episodes_df.iloc[i]
        phases_i = row_i["phase"]
        states_i = row_i["states"]

        for j in range(i + 1, N):
            row_j = episodes_df.iloc[j]
            phases_j = row_j["phase"]
            states_j = row_j["states"]
            dist_ij = 0.0

            for phase_name in all_phases:
                w_phase = phase_weights.get(phase_name, 1.0)
                mask_i = phases_i == phase_name
                mask_j = phases_j == phase_name
                if mask_i.sum() < min_len or mask_j.sum() < min_len:
                    continue

                for joint in joint_names:
                    w_joint = joint_weights.get(joint, 1.0)
                    seq_i = states_i[joint][mask_i]
                    seq_j = states_j[joint][mask_j]
                    d = dtw(seq_i, seq_j)
                    d /= max(len(seq_i), len(seq_j))
                    dist_ij += w_phase * w_joint * d

            D[i, j] = dist_ij
            D[j, i] = dist_ij
        if i % 10 == 0:
            print("iteration", i)

    return D


def get_clusters(
    episodes_df,
    phase_weights=PHASE_WEIGHTS,
    joint_weights=JOINT_WEIGHTS,
    clusters=CLUSTERS,
):
    """Cluster valid episodes and return labels."""
    valid_df = episodes_df.loc[episodes_df["valid"]].reset_index(drop=True)
    D = weighted_episode_dtw_distance(valid_df, phase_weights, joint_weights)
    model = AgglomerativeClustering(
        n_clusters=clusters,
        metric="precomputed",
        linkage="complete",
    )
    return model.fit_predict(D)


def main():
    episodes = create_df()
    labels = get_clusters(episodes)
    print("Cluster labels:", labels)


if __name__ == "__main__":
    main()
