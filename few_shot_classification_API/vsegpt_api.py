from random import choices, seed
from tqdm import tqdm
import pandas as pd
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv('config.env')
api_key = str(os.getenv('API_KEY'))

data = pd.read_csv('Data/USPTO_R_text.csv')
pred_data = data.loc[data["split"] == "test"]
shot_data = data.loc[data["split"] == "train"]

model = 'openai/gpt-oss-120b'

client = OpenAI(api_key=api_key, base_url="https://api.vsegpt.ru/v1")

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
if __name__ == '__main__':
    result = []
    os.makedirs(f'Results/{model}', exist_ok=True)
    for number_of_shots in range(2, 12, 2):
        for random_seed in [36, 42, 84, 200, 12345]:
            for _, test_row in tqdm(pred_data.iterrows(), total=len(pred_data)):
                test_sample_class = "High-yielding" if int(test_row['high_yielding']) == 1 else "Not high-yielding"
                test_sample = test_row['reaction']

                messages = [system_message]
                seed(random_seed)
                for sentences, high_yielding in choices([(row['reaction'], row["high_yielding"])
                                                         for _, row in shot_data.iterrows()],
                                                        k=number_of_shots):

                    high_yielding = "High-yielding" if int(high_yielding) == 1 else "Not high-yielding"

                    messages.append({
                        "role": "user",
                        "content": sentences
                    })
                    messages.append({
                        "role": "assistant",
                        "content": high_yielding
                    })
                messages.append({
                    "role": "user",
                    "content": test_sample
                })

                dirty_response = client.chat.completions.create(model=model, messages=messages)
                llm_answer = dirty_response.choices[0].message.content

                result.append(llm_answer)

            pred_data[f'seed_{random_seed}'] = result
            result.clear()

        pred_data.to_csv(f"Results/{model}/result_shots_{number_of_shots}.csv", index=False)