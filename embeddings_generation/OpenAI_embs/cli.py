import requests
import pandas as pd
from sys import argv
from pathlib import Path
from tqdm import tqdm
import time
import json

API_URL=""

try:
    arg_index = argv.index("--engine")
    if (arg_index + 1) == len(argv):
        print("--engine has been passed, but no value was specified.")
        exit(-1)

    argv.pop(arg_index)
    engine = argv.pop(arg_index)
except ValueError:
    engine = None

if len(argv) == 1:
    print("You need to provide dataset path!")
    exit(-1)

data_path = Path(argv[1])

if engine is None:
    print("Engine was not provided. Using \"text-embedding-3-small\" by default.")
    engine = "text-embedding-3-small"
else:
    print(f"Using engine \"{engine}\".")

output = []
df = pd.read_csv(data_path, sep="\t")
for i, (data,) in tqdm(df.iterrows(), total=df.shape[0]):
    res = requests.get(API_URL + "/respond", data=json.dumps({
        "question": data,
        "engine": engine
    })).json()
    output.append({
        "input": data,
        "embedding": res
    })
    time.sleep(0.3)
json.dump(output, open(f"{argv[1][:-4]}.json", "w", encoding="utf-8"))
