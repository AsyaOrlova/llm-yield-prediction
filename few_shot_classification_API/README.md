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