from __future__ import annotations

import copy
import random
from scipy.spatial import distance
from drfp import DrfpEncoder
import numpy as np

from . import Sampler
from classifier.dataset import DatasetEntry


class TanimotoSampler(Sampler):
    name = "tanimoto"
    __max_length = 5
    __request = None
    __seed = None

    def configure(self, config: dict):
        self.__max_length = config.get("n_for_train", self.__max_length)
        self.__request = config.get("request", self.__request)
        self.__seed = config.get('seed', self.__seed)

    def sample(
        self,
        train: list[DatasetEntry],
        test: list[DatasetEntry],
    ) -> list[DatasetEntry]:

        train = copy.deepcopy(train)
        random.seed(self.__seed)
        random.shuffle(train)
            
        items = []
        request_fp = DrfpEncoder.encode(self.__request, n_folded_length=2048, radius=2)[0]
        
        for i, entry in enumerate(train):
            example_fp = DrfpEncoder.encode(entry.features[0].lstrip('smiles: '), n_folded_length=2048, radius=2)[0]
            similarity = 1 - distance.rogerstanimoto(example_fp, request_fp)
            if similarity >= 0.8:
                items.append(entry)
            if len(items) >= self.__max_length:
                break

        return items
