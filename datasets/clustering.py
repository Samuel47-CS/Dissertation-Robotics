import os
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.signal import medfilt, savgol_filter
from tslearn.metrics import dtw

DATAPATH = "datasets/csv/"
CSV_STRING = "ep{i}.csv"

# Hyper-parameters tuned with Optuna optimiser
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
    "joint_12": 0.7151637734507728
}


# Dataset building and helper functions 

def create_df(datapath=DATAPATH, csv_string=CSV_STRING):
    _, _, files = next(os.walk(datapath))
    num_files = len(files)

    # print(num_files)

    rows = []
    for i in range(num_files):
        f = pd.read_csv(datapath + csv_string.format(i=i))

        # Colect states and timestamps
        states = f.drop(columns=["timestamp"])
        states = pd.DataFrame.from_dict(median_filter_episode(states))
        states = pd.DataFrame.from_dict(savgol_episode(states))
        # states = trim_outliers(states)            <-- Not implemented properly. Required????? Hand picked episode trimming later
        timestamps = f["timestamp"]

        # Calculate pre-grasp, grasp and post-grasp phases
        phases = detect_grasp_phase(states, timestamps)
        phase_array = phases_to_arrayses(phases, len(timestamps))

        if phases is None:
            print("FAILED TO SPLIT PHASES!!! EPISODE", i)

        row = {
            "episode_id": i,
            "n_steps": f.shape[0],
            "duration": float(timestamps[f.shape[0]-1] - timestamps[0]),
            "states": states,
            "timestamps": timestamps,
            "phase": phase_array
        }
        rows.append(row)

    episodes_df = pd.DataFrame(rows)

    return episodes_df


def detect_grasp_phase(states, timestamps,
                       joint_indices=("joint_6", "joint_12"),  # Grippers at joint 6 and 12
                       zero_thresh=3.4,        # threshold to call "near zero"
                       motion_thresh=5.0,      # threshold for "movement" before/after
                       min_duration_s=1):   # optional: min grasp duration in seconds
    """
    states: dict with keys 'timestamps' and 'joint_x' for x = 1, ..., 12
    returns: dict with keys 'grasp_idx'=(start_idx,end_idx),
             'pre_idx' = (0,start-1), 'post_idx'=(end+1, T-1)
             if no valid grasp found, returns None
    """
    j1, j2 = joint_indices
    t = states.shape[0]

    # boolean mask where both are "near zero"
    mask = (np.abs(states[j1]) <= zero_thresh) & (np.abs(states[j2]) <= zero_thresh)

    # find contiguous True segments
    segments = []
    in_seg = False
    for i, val in enumerate(mask):
        if val and not in_seg:
            seg_start = i
            in_seg = True
        elif (not val) and in_seg:
            seg_end = i - 1
            segments.append((seg_start, seg_end))
            in_seg = False
    if in_seg:
        segments.append((seg_start, t-1))

    if not segments:
        return None

    # compute durations and filter by min_duration_s
    segs_valid = []
    for (s,e) in segments:
        dur = timestamps[e] - timestamps[s]
        if dur >= min_duration_s:
            segs_valid.append((s,e,dur))

    if not segs_valid:
        segs_valid = [(s,e,(timestamps[e]-timestamps[s])) for (s,e) in segments]  # fallback

    # pick the longest valid segment
    s,e,dur = max(segs_valid, key=lambda x: x[2])

    # ensure pre and post have motion above motion_thresh
    # check before s: any index i < s where either joint abs > motion_thresh
    has_motion_before = np.any((np.abs(states[j1][:s]) >= motion_thresh) | (np.abs(states[j2][:s]) >= motion_thresh)) if s>0 else False
    has_motion_after  = np.any((np.abs(states[j1][e+1:]) >= motion_thresh) | (np.abs(states[j2][e+1:]) >= motion_thresh)) if e < t-1 else False

    if not (has_motion_before and has_motion_after):
        # try other segments in decreasing duration order to find one that satisfies the condition
        segs_sorted = sorted(segs_valid, key=lambda x: -x[2])
        found = False
        for (s2,e2,dur2) in segs_sorted:
            has_motion_before = np.any((np.abs(states[j1][:s2]) >= motion_thresh) | (np.abs(states[j2][:s2]) >= motion_thresh)) if s2>0 else False
            has_motion_after  = np.any((np.abs(states[j1][e2+1:]) >= motion_thresh) | (np.abs(states[j2][e2+1:]) >= motion_thresh)) if e2 < t-1 else False
            if has_motion_before and has_motion_after:
                s,e = s2,e2
                found = True
                break
        if not found:
            # fallback: choose the longest segment anyway
            pass

    return {"pre_grasp": (0, max(0,int(s)-1)),
            "grasp": (int(s), int(e)),
            "post_grasp": (min(t-1,int(e)+1), t-1)}



def phases_to_arrayses(phase, num_timesteps):
    phase_arr = np.empty(num_timesteps, dtype=object)

    for name, (s, e) in phase.items():
        phase_arr[s:e+1] = name

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
    return np.clip(x, mu - z*sigma, mu + z*sigma)



def remove_bad_episodes(episodes_df, bad_ep_ids):
    episodes_df["valid"] = True
    for ep in bad_ep_ids:
        episodes_df.loc[episodes_df["episode_id"] == ep, "valid"] = False

    print(f"Removed episodes {bad_ep_ids}")
    return episodes_df





# DTW distance / matrix 

def weighted_episode_dtw_distance(
    episodes_df,
    phase_weights = PHASE_WEIGHTS,
    joint_weights = JOINT_WEIGHTS,
    min_len=5):
    """
    Returns:
        D : (N, N) symmetric distance matrix
    """
    N = len(episodes_df)
    D = np.zeros((N, N), dtype=float)

    # infer phase names robustly (union across episodes)
    all_phases = list(set(episodes_df.iloc[0]["phase"]))
    joint_names = list(episodes_df.iloc[0]["states"].keys())

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
                    d = d / max(len(seq_i), len(seq_j))
                    dist_ij += w_phase * w_joint * d

            D[i, j] = dist_ij
            D[j, i] = dist_ij
        if i%10==0:
            print("iteration ", i)

    return D


def get_clusters(episodes_df, 
                 phase_weights = PHASE_WEIGHTS, 
                 joint_weights = JOINT_WEIGHTS,
                 clusters=CLUSTERS):
    
    D = weighted_episode_dtw_distance(
        episodes_df[episodes_df["valid"]],
        phase_weights,
        joint_weights)
    
    model = AgglomerativeClustering(
        n_clusters=clusters,
        metric="precomputed",
        linkage="complete")

    labels = model.fit_predict(D)

    labels_full = np.full(len(episodes_df), -1)
    labels_full[episodes_df["valid"]] = labels

    episodes_df["cluster"] = labels_full
    
    clusters = []
    for i in range(labels.max() + 1):
        C = list(episodes_df[(episodes_df["cluster"] == i)].index)
        R = list(episodes_df[(episodes_df["cluster"] != i)].index)
        clusters.append((C, R))

    return clusters



def main():
    print("creating dataframe")
    episodes_df = create_df()
    print("dataframe created")
    episodes_df = remove_bad_episodes(episodes_df, [8, 145]) # Currently handpicked

    print("Getting clusters")
    clusters = get_clusters(episodes_df)


    for i, cluster in enumerate(clusters):
        print(f"Cluster {i+1}:")
        print(cluster[0])
        print(f"Episodes to be removed from original dataset to create cluster {i+1}:")
        print(cluster[1])
        print()
    return clusters

if __name__ == "__main__":
    main()
