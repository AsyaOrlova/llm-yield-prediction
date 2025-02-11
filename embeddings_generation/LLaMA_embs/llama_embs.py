from langchain_ollama import OllamaEmbeddings
from sys import argv
import pandas as pd
import numpy as np

dataset_path = argv[1]
column = argv[2]

embeddings = OllamaEmbeddings(
    model="llama3.1:8b",
)

df = pd.read_csv(dataset_path)

texts = df[column].tolist()

vectors = embeddings.embed_documents(texts)

np.save(f"result.npy", vectors)