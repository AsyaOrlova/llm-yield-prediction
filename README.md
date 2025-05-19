# Emerging Capacity of Large Language Models to Predict Chemical Reaction Yields

In the ever-expanding field of artificial intelligence, each release of a new large language model (LLM) feels like a drop in the vast ocean of existing models. However, each model possesses distinct characteristics, which often remain underexplored, especially in application to such highly-specialized areas as organic synthesis. In this study, we consider chemical reaction yield prediction as one of the persisting challenges in cheminformatics despite the variety and complexity of machine learning (ML) solutions proposed. We investigate performance and properties of large language models in the context of this complex task. For that, we engineer four chemical reaction datasets, select top-rated generalist LLMs, and systematically evaluate their performance. We demonstrate that Mistral Small and Claude 3 Haiku deliver state-of-the-art performance in few-shot setups, surpassing the strong baselines in accuracy and F1-score. Moreover, we discover superior performance of ML models trained on LLM embeddings and find evidence of yield-relevant information encoded in them. Strikingly, we observe that some general-purpose LLMs outperform those models specifically trained on chemical data. Our findings look promising for the future of chemical language modeling.

![alt text](./images/embs_all.jpg)

## Prerequisites

- For experiments with LLaMA and DeepSeek models [Ollama](https://ollama.com/download) should be installed and running
- Pull a model to use with the library: `ollama pull <model>` e.g. `ollama pull llama3.1:8b`

## :pushpin: Preparation of datasets
The notebooks regarding datasets preparation process can be found in the [datasets_prep_notebooks](./datasets_prep_notebooks) folder. The resulting USPTO-R, USPTO-C, ORD-R and ORD-C datasets as well as USPTO-R based datasets with different train sizes are provided in the [data](./data) folder for your convenience.

## :pushpin: Few-shot classification
The code required to reproduce few-shot classification experiments is provided in the [few_shot_classification_API](./few_shot_classification_API/) (for MistralAI, Anthropic, and OpenAI models) and [few_shot_classification_HF](./few_shot_classification_HF/) (for DeepSeek and MetaAI models) folders.

## :pushpin: Embeddings extraction

The code for extraction of reactions embeddings from text-embedding-3-large, Mistral 7B, and LLaMA-3.1-8B models is provided in [embeddings_generation](./embeddings_generation/) folder.

## :pushpin: Training XGB on DRFPs and LLM embeddings

Scripts for running grid-search and evaluation of XGB models trained on DRFPs and LLM embeddings are provided in the [xgb_drid_search](./xgb_grid_search/) folder.

`python xgb_gs_drfp.py <path_to_smiles_dataset.csv> > result.txt`</br>
`python xgb_gs_embeddings.py <path_to_llm_embeddings_dataset.csv> <path_to_smiles_dataset.csv> > result.txt`
