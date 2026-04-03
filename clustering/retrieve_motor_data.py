"""Tools for extracting motor state data and relabeling dataset metadata."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TASK_TEMPLATE = "Fold the yellow cloth horizontally."
TASK_TEMPLATE_STYLE = "Fold the yellow cloth horizontally. Style: axis 1 {PC0}, axis 2 {PC1}"
SETNAME = "bi-so101-fold-horizontal-set-{set}"
DATA_GET_PATH = Path("data")
CSV_SAVE_PATH = Path("data/csv")


def change_task_descriptions_from_episode(dataset_path, PC_values=None):
    dataset_path = Path(dataset_path)
    info_path = dataset_path / "info.json"

    with open(info_path, "r", encoding="utf-8") as f:
        num_episodes = json.load(f)["total_episodes"]

    for episode in range(num_episodes):
        relabel_tasks(dataset_path / "tasks.jsonl", episode, PC_values)
        relabel_episode_task_descriptions(dataset_path / "episodes.jsonl", episode, PC_values)
        reindex_ep_stats(dataset_path / "episodes_stats.jsonl", episode)


def reindex_ep_stats(filepath, episode=0):
    """Update episode statistics metadata to match the current episode index."""
    filepath = Path(filepath)
    episode = int(episode)

    lines = filepath.read_text(encoding="utf-8").splitlines(keepends=True)
    obj = json.loads(lines[episode])
    stats = obj.setdefault("stats", {})
    task_index = stats.setdefault("task_index", {})
    task_index["min"] = [episode]
    task_index["max"] = [episode]
    task_index["mean"] = [float(episode)]
    lines[episode] = json.dumps(obj) + "\n"
    filepath.write_text("".join(lines), encoding="utf-8")


def relabel_episode_task_descriptions(filepath, episode=0, PC_values=None):
    """Replace a single episode description entry in episodes.jsonl."""
    filepath = Path(filepath)
    episode = int(episode)

    lines = filepath.read_text(encoding="utf-8").splitlines(keepends=True)
    obj = json.loads(lines[episode])
    if PC_values is None:
        obj["tasks"] = [TASK_TEMPLATE.format(ep=episode)]
    else:
        assert len(PC_values) == 2
        obj["tasks"] = [
            TASK_TEMPLATE_STYLE.format(PC0=PC_values[0], PC1=PC_values[1])
        ]
    lines[episode] = json.dumps(obj) + "\n"
    filepath.write_text("".join(lines), encoding="utf-8")


def relabel_tasks(filepath, episode=0, PC_values=None):
    """Relabel the task index and description for a single tasks.jsonl entry."""
    filepath = Path(filepath)
    episode = int(episode)

    lines = filepath.read_text(encoding="utf-8").splitlines(keepends=True)
    obj = json.loads(lines[episode])
    obj["task_index"] = episode
    if PC_values is None:
        obj["task"] = TASK_TEMPLATE.format(ep=episode)
    else:
        assert len(PC_values) == 2
        obj["task"] = TASK_TEMPLATE_STYLE.format(PC0=PC_values[0], PC1=PC_values[1])
    lines[episode] = json.dumps(obj) + "\n"
    filepath.write_text("".join(lines), encoding="utf-8")


def extract_states_and_timestamps(data_get_path, csv_save_path, episode, setnumber=0):
    """Extract motor state and timestamp data from a parquet episode file."""
    data_get_path = Path(data_get_path)
    csv_save_path = Path(csv_save_path)
    episode = int(episode)
    episode_id = episode + 10 * setnumber
    episode_str = str(episode).zfill(3)

    parquet_path = data_get_path / "data" / "chunk-000" / f"file-{episode_str}.parquet"
    df = pd.read_parquet(parquet_path)

    state_column = None
    for candidate in ["observation.state", "observation", "observation_state"]:
        if candidate in df.columns:
            state_column = candidate
            break
    if state_column is None:
        state_column = next((c for c in df.columns if c.startswith("observation")), None)
    if state_column is None:
        raise KeyError(
            f"No observation state column found in {parquet_path}. "
            f"Available columns: {list(df.columns)}"
        )
    
    print(state_column)

    observations = df[state_column]
    if observations.dtype == object and isinstance(observations.iloc[0], dict):
        states = np.vstack([
            np.asarray(s["state"]) if isinstance(s, dict) and "state" in s else np.asarray(s)
            for s in observations
        ])
    else:
        states = np.vstack([np.asarray(s) for s in observations])

    timestamps = np.array(df["timestamp"]).reshape(-1, 1)
    data = np.hstack((timestamps, states))
    columns = ["timestamp"] + [f"joint_{i+1}" for i in range(states.shape[1])]
    rdf = pd.DataFrame(data, columns=columns)
    output_path = csv_save_path / f"ep{episode_id}.csv"
    rdf.to_csv(output_path, index=False)


def extract_states_and_timestamps_from_df(df, csv_save_path, episode, setnumber=0):
    """Extract motor state and timestamp data for a single episode from the loaded dataframe."""
    csv_save_path = Path(csv_save_path)
    episode = int(episode)
    episode_id = episode + 10 * setnumber

    # Filter dataframe for this episode
    episode_df = df[df["episode_index"] == episode]

    state_column = None
    for candidate in ["observation.state", "observation", "observation_state"]:
        if candidate in episode_df.columns:
            state_column = candidate
            break
    if state_column is None:
        state_column = next((c for c in episode_df.columns if c.startswith("observation")), None)
    if state_column is None:
        raise KeyError(
            f"No observation state column found. "
            f"Available columns: {list(episode_df.columns)}"
        )

    observations = episode_df[state_column]
    if observations.dtype == object and len(observations) > 0 and isinstance(observations.iloc[0], dict):
        states = np.vstack([
            np.asarray(s["state"]) if isinstance(s, dict) and "state" in s else np.asarray(s)
            for s in observations
        ])
    else:
        states = np.vstack([np.asarray(s) for s in observations])

    timestamps = np.array(episode_df["timestamp"]).reshape(-1, 1)
    data = np.hstack((timestamps, states))
    columns = ["timestamp"] + [f"joint_{i+1}" for i in range(states.shape[1])]
    rdf = pd.DataFrame(data, columns=columns)
    output_path = csv_save_path / f"ep{episode_id}.csv"
    rdf.to_csv(output_path, index=False)


def extract_all_data(data_get_path=DATA_GET_PATH, csv_save_path=CSV_SAVE_PATH, setname=None):
    """Extract episode CSV files from the dataset directory structure."""
    data_get_path = Path(data_get_path)
    csv_save_path = Path(csv_save_path)
    csv_save_path.mkdir(parents=True, exist_ok=True)

    if setname is None:
        dataset_dirs = sorted(
            p.parent
            for p in data_get_path.glob("bi-so101-fold-horizontal-set-*/meta/info.json")
        )
    else:
        dataset_dirs = [data_get_path / SETNAME.format(set=setname)]

    for dataset_dir in dataset_dirs:
        info_path = dataset_dir / "meta" / "info.json"
        with open(info_path, "r", encoding="utf-8") as f:
            num_episodes = json.load(f)["total_episodes"]

        # Load the entire dataset
        parquet_path = dataset_dir / "data" / "chunk-000" / "file-000.parquet"
        df = pd.read_parquet(parquet_path)

        for episode in range(num_episodes):
            print(
                f"Extracting motor control data from episode {episode} in {dataset_dir.name}"
            )
            extract_states_and_timestamps_from_df(df, csv_save_path, episode)


def main():
    parser = argparse.ArgumentParser(
        description="Extract CSV files from dataset episodes."
    )
    parser.add_argument(
        "--setname",
        default="full-v3",
        help="Set name to extract, e.g. full-v3 or 1.",
    )
    args = parser.parse_args()
    extract_all_data(setname=args.setname)


if __name__ == "__main__":
    main()