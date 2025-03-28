from langchain_core.prompts import ChatPromptTemplate, FewShotPromptTemplate, PromptTemplate
from langchain_ollama.llms import OllamaLLM
# from langchain.chains import FewShotChain
import pandas as pd
from tqdm import tqdm
from sys import argv
from scipy.spatial import distance
from drfp import DrfpEncoder

dataset_path = argv[1]
column = argv[2]

df = pd.read_csv(dataset_path)

df['high_yielding'] = df['high_yielding'].apply(lambda x: 'High-yielding' if x==1 else 'Not high-yielding')

df_train = pd.DataFrame(df[df.split=='train'])
df_test = pd.DataFrame(df[df.split=='test'])


zero_shot_template = {
    "smiles": "You are an expert chemist. Your task is to predict reaction yields based on SMILES representations"
              " of organic reactions. Reaction SMILES consist of potentially three parts (reactants, agents,"
              " and products) each separated by an arrow symbol '>'. Reactants are listed before the arrow symbol."
              " If a reaction includes agents, such as catalysts or solvents, they can be included after the reactants."
              " Products are listed after the second arrow symbol, representing the resulting substances of the"
              " reaction. You can only predict whether the reaction is 'High-yielding' or 'Not high-yielding'."
              " 'High-yielding' reaction means the yield rate of the reaction is above 70%. 'Not high-yielding'"
              " means the yield rate of the reaction is below 70%. You will be provided with several examples of"
              " reactions and corresponding yield rates. Strictly answer with only 'High-yielding' or"
              " 'Not high-yielding', no other information can be provided.",
    "text": "You are an expert chemist. Based on text descriptions of organic reactions"
            " you predict their yields using your experienced reaction yield prediction knowledge."
            " You can only predict whether the reaction is 'High-yielding' or 'Not high-yielding'."
            " 'High-yielding' reaction means the yield rate of the reaction is above 70%."
            " 'Not high-yielding' means the yield rate of the reaction is below 70%."
            " You will be provided with several examples of reactions and corresponding yield rates."
            " Please answer with only 'High-yielding' or 'Not high-yielding', no other information can be provided."}

# k = 6
# seed = 36
system_prompt = zero_shot_template['text']
model = OllamaLLM(model="llama3.3:70b", temperature=0.5, num_predict=5)
example_prompt = PromptTemplate.from_template("Reaction: {question}\nAnswer: {answer}")
instruction = zero_shot_template['smiles']

# prompt_template = PromptTemplate(input_variables=['examples', 'reaction'],
#                                  template="You are an expert chemist. Based on text descriptions of organic reactions"
#                                  " you predict their yields using your experienced reaction yield prediction knowledge."
#                                   " You can only predict whether the reaction is 'High-yielding' or 'Not high-yielding'."
#                                   " 'High-yielding' reaction means the yield rate of the reaction is above 70%."
#                                   " 'Not high-yielding' means the yield rate of the reaction is below 70%."
#                                   " You will be provided with several examples of reactions and corresponding yield rates."
#                                   " Please answer with only 'High-yielding' or 'Not high-yielding', no other information can be provided.\n"
#                                   "{examples}\Reaction: {reaction}"
#                                  )


# def equal_sampler(seed, k):
  
#   samples = []
  
#   for idx in df_train.index:
#     d = dict()
#     if not samples:
#       d['question'] = df_train.loc[idx, column]
#       d['answer'] = df_train.loc[idx, 'high_yielding']
#       samples.append(d)
#       df_train.drop([idx], inplace=True)
#     else:
#       if df_train.loc[idx, 'high_yielding'] not in samples[0].values():
#         d['question'] = df_train.loc[idx, column]
#         d['answer'] = df_train.loc[idx, 'high_yielding']
#         samples.append(d)
#         df_train.drop([idx], inplace=True)
#         break
   
#   df_choice = df_train.sample(k-2, random_state=seed)
#   for idx in df_choice.index:
#     d = dict()
#     d['question'] = df_train.loc[idx, column]
#     d['answer'] = df_train.loc[idx, 'high_yielding']
#     samples.append(d)
  
#   return samples


def tanimoto_sampler(df_train, request, seed, k):
  
  df_train = df_train.sample(frac=1, random_state=seed).reset_index(drop=True)
  
  items = []
  request_fp = DrfpEncoder.encode(request, n_folded_length=2048, radius=2)[0]
  
  for idx in df_train.index:
    d = dict()
    entry = df_train.loc[idx, column]
    cl = df_train.loc[idx, 'high_yielding']
    example_fp = DrfpEncoder.encode(entry, n_folded_length=2048, radius=2)[0]
    similarity = 1 - distance.rogerstanimoto(example_fp, request_fp)
    if similarity >= 0.8:
        d['question'] = df_train.loc[idx, column]
        d['answer'] = df_train.loc[idx, 'high_yielding']
        items.append(d)
    if len(items) >= k:
        break
  
  return items

# def equal_sampler(seed, k):
  
#   samples = []
  
#   for idx in df_train.index:
#     d = dict()
#     if not samples:
#       d['question'] = df_train.loc[idx, column]
#       d['answer'] = df_train.loc[idx, 'high_yielding']
#       samples.append(d)
#       df_train.drop([idx], inplace=True)
#     else:
#       if df_train.loc[idx, 'high_yielding'] not in samples[0].values():
#         d['question'] = df_train.loc[idx, column]
#         d['answer'] = df_train.loc[idx, 'high_yielding']
#         samples.append(d)
#         df_train.drop([idx], inplace=True)
#         break
   
#   df_choice = df_train.sample(k-2, random_state=seed)
#   for idx in df_choice.index:
#     d = dict()
#     d['question'] = df_train.loc[idx, column]
#     d['answer'] = df_train.loc[idx, 'high_yielding']
#     samples.append(d)
  
#   new_samples = []
#   for i in samples:
#     text = f"Reaction: {i['question']}\nAnswer: {i['answer']}"
#     new_samples.append(text)
  
#   return new_samples

# def get_response(system_prompt, samples, reaction, model, seed):
#   messages = [{"role": "system", "content": system_prompt}]
#   for s, y in samples.items():
#     y = 'high_yielding' if y==1 else 'not_high_yielding'
#     messages.append(
#                 {"role": "user", "content": s}
#             )
#     messages.append({"role": "assistant", "content": y})  
#   messages.append({"role": "user", "content": reaction}) 
#   response: ChatResponse = chat(model=model, messages=messages, options={'seed':seed})
#   return response['message']['content']


for k in tqdm([2, 4, 6, 8, 10]):
  for seed in [36, 42, 84, 200, 12345]:
    for idx in tqdm(df_test.index):
        reaction = df_test.loc[idx, column]
        # print(reaction)
        samples = tanimoto_sampler(df_train, reaction, seed, k)
        print(samples)
        # few_shot_chain = FewShotChain(llm=model, prompt=prompt_template, support=samples)
        # for i in samples:
        #   print(i)
        #   print('\n')
        prompt = FewShotPromptTemplate(examples=samples, example_prompt=example_prompt, suffix="Reaction: {input}\nAnswer:", input_variables=["input"], prefix=instruction)
        # print(prompt)
        # final_prompt = prompt.invoke({"input": reaction}).to_string()
        # print(final_prompt)
        chain = prompt | model
        out = chain.invoke({"input": reaction})
        # out = few_shot_chain.run(query=reaction)
        print(out)
        df_test.loc[idx, f'response_{k}_{seed}'] = out
  df_test.to_csv(f'{dataset_path}_llama33_tanimoto_results.csv', index=False)

df_test.to_csv(f'{dataset_path}_llama33_tanimoto_results.csv', index=False)
  