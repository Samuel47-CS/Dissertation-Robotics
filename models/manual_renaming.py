import os
import pandas as pd
from safetensors import safe_open
from safetensors.torch import save_file

MODELS_DIR = 'models/'
TARGET = 'wrist.right'
REPLACEMENT = 'right_wrist'

def process_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    if TARGET in data:
        data = data.replace(TARGET, REPLACEMENT)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(data)

def process_parquet(file_path):
    df = pd.read_parquet(file_path)
    df = df.map(lambda x: x.replace(TARGET, REPLACEMENT) if isinstance(x, str) and TARGET in x else x)
    df.to_parquet(file_path)

def process_safetensor(file_path):
    with safe_open(file_path, framework="pt", device="cpu") as f:
        tensors = {k: f.get_tensor(k) for k in f.keys()}
        metadata = f.metadata()  

    # Rename tensor keys if they contain TARGET
    new_tensors = {}
    for k, v in tensors.items():
        new_k = k.replace(TARGET, REPLACEMENT)
        new_tensors[new_k] = v

    # Replace in metadata strings if they contain TARGET
    new_metadata = {}
    if metadata:
        for k, v in metadata.items():
            new_k = k.replace(TARGET, REPLACEMENT)
            if isinstance(v, str):
                new_v = v.replace(TARGET, REPLACEMENT)
            else:
                new_v = v
            new_metadata[new_k] = new_v
    else:
        new_metadata = None

    # Save back with renamed tensors and updated metadata
    save_file(new_tensors, file_path, metadata=new_metadata)

if __name__ == '__main__':

    print(f"Renaming occurrance of '{TARGET}' to '{REPLACEMENT}'")

    for model_dir in os.listdir(MODELS_DIR):
        model_path = os.path.join(MODELS_DIR, model_dir)
        if os.path.isdir(model_path):
            print(model_dir)
            for root, _, files in os.walk(model_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if file.endswith('.json'):
                        process_json(file_path)
                    elif file.endswith('.parquet'):
                        process_parquet(file_path)
                    elif file.endswith('.safetensors'):
                        process_safetensor(file_path)