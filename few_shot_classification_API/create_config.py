configs = {
    "sampler": ["equal"],
    "anthropic": ["claude-3-opus-20240229", "claude-3-haiku-20240307"],
    "openai": ["gpt-3.5-turbo", "gpt-4"],
    "mistral": ["mistral-small-latest", "mistral-large-latest"],
    "n_for_train": [2, 4, 6, 8, 10],
    "seed": [36, 42, 84, 200, 12345]
}
c = 1
for dataset in ['../data/USPTO_R_text.csv']:
    for sampler in ["equal"]:
        for provider in ["anthropic"]:
            for engine in configs[provider]:
                for n in configs["n_for_train"]:
                    for seed in configs["seed"]:
                        with open(f"./{c}.txt", "w", encoding="utf-8") as file:
                            file.write(f"name=name\n"
                                    f"subject=chemistry\n"
                                    f"provider={provider}\n"
                                    f"engine={engine}\n"
                                    f"sampler={sampler}\n"
                                    f"dataset={dataset}\n"
                                    f"data_format=table\n"
                                    f"classes=high_yielding,not_high_yielding\n"
                                    f"n_for_train={n}\n"
                                    f"test_size=0.3\n"
                                    f"seed={seed}\n"
                                    f"enable_metrics=True"
                                    )
                            c += 1