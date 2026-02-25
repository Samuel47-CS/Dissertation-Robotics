import pandas as pd
import numpy as np
import json
import os

TASK_TEMPLATE = "Fold the yellow cloth horizontally."
# TASK_TEMPLATE = "Fold the yellow cloth horizontally. EP{ep}"
TASK_TEMPLATE_STYLE = "Fold the yellow cloth horizontally. Style: axis 1 {PC0}, axis 2 {PC1}"
SETNAME = "bi-so101-fold-horizontal-set-{set}/"
DATA_GET_PATH = "datasets/"
CSV_SAVE_PATH= "datasets/csv/"


# --------------- Relabelling -----------------

def change_task_descriptions_from_episode(dataset_path, PC_values=None):

    infopath = dataset_path + "info.json"

    with open(infopath, "r") as f:
        num_episodes = json.load(f)["total_episodes"]

    for episode in range(num_episodes):
        relabel_tasks(dataset_path + "tasks.jsonl", episode, PC_values)
        relabel_episode_task_descriptions(dataset_path + "episodes.jsonl", episode, PC_values)
        reindex_ep_stats(dataset_path + "episodes_stats.jsonl", episode)


def reindex_ep_stats(filepath, episode=0):
    '''
    Replaced task indices in episode statistic metadata with task index
    corresponding to the episode number in episode_stats.jsonl
    
    :param filepath: File path from top level (Robotics directory) into relevant 
    datasets meta data episode statistics
    :param episode: Episode index
    '''
    if type(episode) is not int:
        episode = int(episode)

    with open(filepath, "r") as f:
        lines = f.readlines()

    obj = json.loads(lines[episode])
    obj["stats"]["task_index"]["min"] = [episode]
    obj["stats"]["task_index"]["max"] = [episode]
    obj["stats"]["task_index"]["mean"] = [float(episode)]
    
    lines[episode] = json.dumps(obj) + "\n"

    with open(filepath, "w") as f:
        f.writelines(lines)

def relabel_episode_task_descriptions(filepath, episode=0, PC_values=None):
    '''
    Replace task desciption corresponding with the episode in episode.jsonl
    
    :param filepath: File path from top level (Robotics directory) into relevant 
    datasets meta data episode description
    :param episode: Episode index
    :param PC_values: Values aqcuired from PCA
    '''
    if type(episode) is not int:
        episode = int(episode)

    with open(filepath, "r") as f:
        lines = f.readlines()

    obj = json.loads(lines[episode])
    if PC_values is None:
        obj["tasks"] = [TASK_TEMPLATE.format(ep=episode)]
    else:
        assert len(PC_values) == 2
        obj["tasks"] = [TASK_TEMPLATE_STYLE.format(PC0=PC_values[0], PC1=PC_values[1])]

    lines[episode] = json.dumps(obj) + "\n"

    with open(filepath, "w") as f:
        f.writelines(lines)


def relabel_tasks(filepath, episode=0, PC_values=None):
    '''
    Relabels task indices and task descriptions based on principal components
    in tasks.jsonl
    
    :param filepath: File path from top level (Robotics directory) into relevant 
    datasets meta data task description
    :param episode: Episode index
    :param PC_values: Values aqcuired from PCA
    '''
    if type(episode) is not int:
        episode = int(episode)

    with open(filepath, "r") as f:
        lines = f.readlines()

    obj = json.loads(lines[episode])
    obj["task_index"] = episode
    if PC_values is None:
        obj["task"] = TASK_TEMPLATE.format(ep=episode)
    else:
        assert len(PC_values) == 2
        obj["task"] = TASK_TEMPLATE_STYLE.format(PC0=PC_values[0], PC1=PC_values[1])

    lines[episode] = json.dumps(obj) + "\n"

    with open(filepath, "w") as f:
        f.writelines(lines)


# --------------- Data Extraction -----------------

def extract_states_and_timestamps(data_get_path, csv_save_path, episode=None, setnumber=0):
    '''
    Saves data in (X x 13) shape, where first column is timestamps, and the 
    next 12 are motor controls corresponding to the timestamps
    
    :param data_get_path: Path to episode data
    :param csv_save_path: Path to csv directory 
    :param episode: Episode index
    '''
    assert episode is not None
    if type(episode) is int:
        episode_id = str(episode + 10*setnumber)
        episode = str(episode)
    else:
        episode_id = str(int(episode) + 10*setnumber)
        

    df = pd.read_parquet(data_get_path+f'data/chunk-000/episode_{episode.zfill(6)}.parquet')
    states = np.vstack([np.asarray(s) for s in df["observation.state"]])
    timestamps = np.array(df["timestamp"]).reshape(-1, 1)
    data = np.hstack((timestamps, states))
    rdf = pd.DataFrame(data, columns=["timestamp"] + [f"joint_{i+1}" for i in range(states.shape[1])])
    rdf.to_csv(csv_save_path + f'ep{episode_id}.csv', index=False)



def extract_all_data(data_get_path=DATA_GET_PATH, csv_save_path=CSV_SAVE_PATH, setname=None):
    '''
    Clips motor control sequence data into dataset of timestamps and motor 
    positions. Saves data into individual .csv files based on episode into 
    .csv directory.
    
    :param data_get_path: Path to episode data
    :param csv_save_path: Path to csv directory 

    TODO: Source 
    '''
    

    if setname is None:

        num_sets = len(os.listdir(data_get_path))
        for i in range(num_sets):
            filepath = data_get_path + SETNAME.format(set=i+1) + "meta/info.json"
            
            with open(filepath, "r") as f:
                num_episodes = json.load(f)["total_episodes"]

            for j in range(num_episodes):
                print(f"Extracting motor control data from episode {j + i*10}") # TODO: probably change the display here
                extract_states_and_timestamps(data_get_path + SETNAME.format(set=i+1), csv_save_path, j, i)


    else:
        filepath = data_get_path + SETNAME.format(set=setname) + "meta/info.json"

        with open(filepath, "r") as f:
            num_episodes = json.load(f)["total_episodes"]

        for j in range(num_episodes):
            print(f"Extracting motor control data from episode {j}") # TODO: probably change the display here
            extract_states_and_timestamps(data_get_path + SETNAME.format(set=setname), csv_save_path, j)




def main():
    extract_all_data(setname="full")
    # change_task_descriptions_from_episode(DATA_GET_PATH + "bi-so101-fold-horizontal-set-full/" + "meta/")

if __name__ == "__main__":
    main()