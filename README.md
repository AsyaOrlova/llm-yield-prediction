# Ocean in a Drop: Underexplored Capabilities of Large Language Models in Chemistry

In the ever-expanding field of artificial intelligence, each release of a new large language model (LLM) feels like a drop in the vast ocean of existing models. However, each model possesses distinct characteristics, which often remain underexplored, especially in application to such highly-specialized areas as chemistry. In this study, we consider chemical reaction yield prediction as one of the persisting challenges in cheminformatics despite the variety and complexity of machine learning (ML) solutions proposed. We investigate performance and properties of large language models in the context of this complex task. For that, we engineer four different chemical reaction datasets, select top-rated generalist LLMs, and systematically evaluate their performance. We demonstrate that Mistral Small and Claude 3 Haiku, whose significance has been diminished with emergence of newer models, systematically deliver state-of-the-art performance in few-shot setups, surpassing baselines by up to 3% in accuracy and F1-score. Moreover, we discover superior performance of ML models trained on LLM embeddings and find evidence of yield-relevant information encoded in them. Strikingly, we observe that some general-purpose LLMs outperform those models specifically trained on chemical data. These findings allude to the number of underexplored properties of an individual LLM, as an ocean in the drop.

![alt text](./images/embs_all.jpg)

## :pushpin: Preparation of datasets
The notebooks regarding datasets preparation process can be found in the [datasets_prep_notebooks](./datasets_prep_notebooks) folder. The resulting USPTO-R, USPTO-C, ORD-R and ORD-C datasets as well as USPTO-R based datasets with different train sizes are provided in the [data](./data) folder for your convenience.

## :pushpin: Few-shot classification
The code needed to reproduce few-shot classification experiments is provided in the [few_shot_classifier](./few_shot_classifier/) folder.

### User guide

`pip install requirements.txt`</br>
`python -m classifier <path_to_config_file>`

### Config file structure

**name=name**  - any identifier for your experiments folder</br>
**subject=chemistry** - area of your experiments</br>
**provider=openai** - provider (openai, mistral, anthropic are available)</br>
**engine=gpt-4** - LLM name</br>
**dataset=./data/dataset.csv** - path to the dataset</br>
**data_format=text** - data format ("text" if your data is in the format of sentences, "table" if your data is in the format of features)</br>
**classes=class_1,class_2** - classes of your data (the same as in your dataset, comma-separated, without spaces)</br>
**n_for_train=7** - number of examples in prompt</br>
**seed=36** - random seed needed to obtain reproducible results</br>
**enable_metrics=True** - whether to calculate metrics for the experiment or not</br>

Example of the config file is provided in the folder.

If you want to run multiple experiments in one pipeline, you can create configs with [create_config.py](./few_shot_classifier/create_config.py), then run the [run_commands.py](./few_shot_classifier/run_commands.py) and gather the obtained experimental parameters and corresponding metrics using the [gather_metrics.py](./few_shot_classifier/gather_metrics.py).

## :pushpin: Analysis of few-shot results
The notebooks with the few-shot results analysis are provided in the [few_shot_notebooks](./few_shot_notebooks/) folder.  We also provide .csv files with metrics for different experiments, however you can obtain them yourself by reproducing the few-shot experiments as suggested in the previous section.

## :pushpin: LLM embeddings extraction

The code for extraction of reactions embeddings from text-embedding-3-large (OpenAI) and Mistral 7B (MistralAI) is provided in [openai_emb_gen](./openai_emb_gen/) and [mistralai_emb_gen](./mistralai_emb_gen/) folders.

### OpenAI embeddings extraction

1. The dataset should contain only one column without indices.
2. `pip install -r requirements.txt`
3. `python cli.py <path_to_dataset.csv> --engine <engine_name>` (text-embedding-3-large and text-embedding-3-small are available)
4. Output in result.json

### Mistral 7B embeddings extraction

1. `pip install -r requirements.txt`
2. `python cli.py <path_to_dataset.csv>`
3. Output in result.json.

### LLaMA experiments

The code for extraction of LLaMA embeddings and few-shot experiments is provided in [llama_exp](./llama_exp/).

Examples:

`python llama_embs.py ./data/USPTO_R_text.csv reaction`

`python llama_fewshot.py ./data/USPTO_R_text.csv reaction llama3.1:8b USPTO_R_results.csv text`

## :pushpin: Training XGB on DRFPs and LLM embeddings

Scripts for running grid-search and evaluation of XGB models trained on DRFPs and LLM embeddings are provided in the [xgb_drid_search](./xgb_grid_search/) folder.

`python xgb_gs_drfp.py <path_to_smiles_dataset.csv> > result.txt`</br>
`python xgb_gs_embeddings.py <path_to_llm_embeddings_dataset.csv> <path_to_smiles_dataset.csv> > result.txt`
