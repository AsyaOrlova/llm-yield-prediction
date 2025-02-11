from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_ollama.llms import OllamaLLM
import pandas as pd
from tqdm import tqdm
import argparse
from scipy.spatial import distance
from drfp import DrfpEncoder


instruction_template = {
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
            " Please answer with only 'High-yielding' or 'Not high-yielding', no other information can be provided."
            }



def random_sampler(df_train, seed, k, column):
  
  """"Samples at least one example form each catgory."""
  
  samples = []
  
  for idx in df_train.index:
    d = dict()
    if not samples:
      d['question'] = df_train.loc[idx, column]
      d['answer'] = df_train.loc[idx, 'high_yielding']
      samples.append(d)
      df_train.drop([idx], inplace=True)
    else:
      if df_train.loc[idx, 'high_yielding'] not in samples[0].values():
        d['question'] = df_train.loc[idx, column]
        d['answer'] = df_train.loc[idx, 'high_yielding']
        samples.append(d)
        df_train.drop([idx], inplace=True)
        break
   
  df_choice = df_train.sample(k-2, random_state=seed)
  for idx in df_choice.index:
    d = dict()
    d['question'] = df_train.loc[idx, column]
    d['answer'] = df_train.loc[idx, 'high_yielding']
    samples.append(d)
  
  return samples


def tanimoto_sampler(df_train, request, seed, k, column):
  
  """"Samples top-N examples according to the Tanimoto similarity with the target reaction."""
  
  df_train = df_train.sample(frac=1, random_state=seed).reset_index(drop=True)
  
  items = []
  request_fp = DrfpEncoder.encode(request, n_folded_length=2048, radius=2)[0]
  
  for idx in df_train.index:
    d = dict()
    entry = df_train.loc[idx, column]
    example_fp = DrfpEncoder.encode(entry, n_folded_length=2048, radius=2)[0]
    similarity = 1 - distance.rogerstanimoto(example_fp, request_fp)
    if similarity >= 0.8:
        d['question'] = df_train.loc[idx, column]
        d['answer'] = df_train.loc[idx, 'high_yielding']
        items.append(d)
    if len(items) >= k:
        break
  
  return items


def parameters():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path', type=str, default='../data/USPTO_R_text.csv')
    parser.add_argument('--column', type=str, default='reaction')
    parser.add_argument('--model', type=str, default='llama3.1:8b')
    parser.add_argument('--format', type=str, default='text', choices=['smiles', 'text'])
    parser.add_argument('--sampler', type=str, default='random', choices=['random', 'tanimoto'])
    return parser.parse_args()


def main():
    # parse parameters
    args = vars(parameters())
    data_path = args['data_path']
    column = args['column']
    model = args['model']
    format = args['format']
    sampler = args['sampler']
    
    # prepare data
    df = pd.read_csv(data_path)
    df['high_yielding'] = df['high_yielding'].apply(lambda x: 'High-yielding' if x==1 else 'Not high-yielding')
    df_train = pd.DataFrame(df[df.split=='train'])
    df_test = pd.DataFrame(df[df.split=='test'])
    
    # prepare prompt template and load model
    example_prompt = PromptTemplate.from_template("Reaction: {question}\nAnswer: {answer}")
    model = OllamaLLM(model=model, temperature=0.5, num_predict=5)    
    
    for k in tqdm([2, 4, 6, 8, 10]):
      for seed in [36, 42, 84, 200, 12345]:
        for idx in tqdm(df_test.index):

            # identify target reaction
            reaction = df_test.loc[idx, column]
            
            # sample k examples from training set
            if sampler == 'random':
              samples = random_sampler(df_train, seed, k, column)
            else:
              samples = tanimoto_sampler(df_train, reaction, seed, k, column)
            
            # create few-shot prompt and send it to the model
            prompt = FewShotPromptTemplate(examples=samples,
                                           example_prompt=example_prompt,
                                           suffix="Reaction: {input}\nAnswer:",
                                           input_variables=["input"],
                                           prefix=instruction_template[format])
            chain = prompt | model
            
            # parse the response
            out = chain.invoke({"input": reaction})
            df_test.loc[idx, f'response_{k}_{seed}'] = out

    # save dataset with answers
    df_test.to_csv(f'{data_path[:-4]}_{model}_result.csv', index=False)
    
    
if __name__ == "__main__":
    main()