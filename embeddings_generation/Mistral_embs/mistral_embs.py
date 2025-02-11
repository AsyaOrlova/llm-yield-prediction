import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
import json
import tqdm
from sys import argv

dataset_path = argv[1]
column = argv[2]

df = pd.read_csv(dataset_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with torch.no_grad():
  model = AutoModel.from_pretrained('mistralai/Mistral-7B-v0.1', torch_dtype=torch.float16).to(device)
  tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1')
  results = []
  for idx in tqdm.tqdm(df.index):
    reaction = df.loc[idx, column]
    enc = tokenizer(reaction, return_tensors="pt", truncation=True, max_length=512).to(device)
    emb = model(**enc)[0].last_hidden_state.cpu()[0][-1]
    results.append({
           "input": reaction,
           "embedding": emb.tolist()
   })

with open("result.json", "w") as file:
   json.dump(
       list(results),
       file)