import subprocess
import time
from ollama import Client
from sys import argv
import pandas as pd
from random import choices, seed
from tqdm import tqdm

data_path = argv[1]
data = pd.read_csv(data_path)
pred_data = data.loc[data["split"] == "test"]
shot_data = data.loc[data["split"] == "train"]

# ollama start up
model = argv[2]

column = argv[3]

print("="*50, "launching OLLAMA", "="*50)
server_process = subprocess.Popen(['ollama', 'serve'])
time.sleep(10)
print("="*50, "OLLAMA is ready", "="*50)
# client sync

client = Client(host='host')

# download model if it does not exist
print("="*50, "downloading model", "="*50)
client.pull(model)
client.create(model=f"{model.split(':')[0]}-t1", parameters={"temperature": 0.1}, from_=model)
print("="*50, "model downloaded", "="*50)

# system instructions
system_message = {
    'role': 'system',
    'content': "You are an expert chemist. Based on text descriptions of organic reactions"
                 " you predict their yields using your experienced reaction yield prediction knowledge."
                 " You can only predict whether the reaction is 'High-yielding' or 'Not high-yielding'."
                 " 'High-yielding' reaction means the yield rate of the reaction is above 70%."
                 " 'Not high-yielding' means the yield rate of the reaction is below 70%."
                 " You will be provided with several examples of reactions and corresponding yield rates."
                 " Please answer with only 'High-yielding' or 'Not high-yielding', no other information can be "
                 "provided."
}

print("="*50, "start PREDICTION", "="*50)
result = []
for ns in [i * 2 for i in range(5, 6, 1)]:
    for sd in [36, 42, 84, 200, 12345]:
        for pred_sample in tqdm(pred_data.iterrows(), total=len(pred_data)):
            messages = [system_message]
            seed(sd)
            for sentences, high_yielding in choices(list(map(lambda x: (x[1][column], x[1]["high_yielding"]),
                                                             shot_data.iterrows())), k=ns):
                messages.append({
                    "role": "user",
                    "content": sentences
                })
                messages.append({
                    "role": "assistant",
                    "content": "high-yielding" if int(high_yielding) == 1 else "not high-yielding"
                })

            messages.append({
                "role": "user",
                "content": pred_sample[1][column]
            })

            response = client.chat(model=f"{model.split(':')[0]}-t1", messages=messages)
            result.append((pred_sample[1][column], response.message.content.split("</think>")[-1].strip()))
        with open(f"result_shots{ns}_seed{sd}.txt", "w", encoding="utf-8") as file:
            for s, r in result:
                file.write(f"{s.replace('\n', '\\n')}\t{r.replace('\n', '\\n')}\n")
        result.clear()

