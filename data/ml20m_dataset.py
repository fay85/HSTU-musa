"""MovieLens 20M dataset for HSTU training.

Reads processed_seqs.csv (produced by recsys-examples hstu_data_preprocessor)
and yields HSTUBatch objects compatible with the MUSA HSTU model.

CSV format: user_id, movie_id (JSON list), rating (JSON list), unix_timestamp (JSON list)
"""

import json
import math
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset

from compat.jagged_tensor import JaggedTensor, KeyedJaggedTensor
from data.batch import HSTUBatch


def load_seq(x):
    if isinstance(x, str):
        return json.loads(x)
    return x


def maybe_truncate_seq(y: List[int], max_len: int) -> List[int]:
    return y[:max_len] if len(y) > max_len else y


class ML20MDataset(IterableDataset):
    """
    MovieLens 20M sequence dataset for HSTU.

    Matches the original recsys-examples data pipeline:
    - user_id as contextual feature (single token per sample)
    - movie_id as item feature (variable-length sequence)
    - rating as action feature (variable-length sequence, aligned with movie_id)
    - Last max_num_candidates items are ranking candidates; rest is history
    - 70/30 train/test split
    """

    ITEM_FEATURE = "movie_id"
    ACTION_FEATURE = "rating"
    CONTEXTUAL_FEATURES = ["user_id"]

    def __init__(
        self,
        csv_path: str,
        batch_size: int = 128,
        max_history_seqlen: int = 200,
        max_num_candidates: int = 20,
        num_tasks: int = 1,
        is_train: bool = True,
        shuffle: bool = False,
        random_seed: int = 1234,
        nrows: Optional[int] = None,
    ) -> None:
        super().__init__()
        df = pd.read_csv(csv_path, nrows=nrows)

        if max_num_candidates > 0:
            valid_mask = df[self.ITEM_FEATURE].apply(
                lambda x: len(load_seq(x)) > max_num_candidates
            )
            df = df[valid_mask].reset_index(drop=True)

        n_total = len(df)
        n_train = int(n_total * 0.7)
        if is_train:
            df = df.head(n_train)
        else:
            df = df.tail(n_total - n_train)

        self._df = df.reset_index(drop=True)
        self._num_samples = len(self._df)
        self._batch_size = batch_size
        self._max_history_seqlen = max_history_seqlen
        self._max_num_candidates = max_num_candidates
        self._num_tasks = num_tasks
        self._shuffle = shuffle
        self._random_seed = random_seed

        self._feature_to_max_seqlen = {
            **{name: 1 for name in self.CONTEXTUAL_FEATURES},
            self.ITEM_FEATURE: max_history_seqlen + max_num_candidates,
            self.ACTION_FEATURE: max_history_seqlen + max_num_candidates,
        }

        self._sample_ids = np.arange(self._num_samples)
        self._do_shuffle()

    def _do_shuffle(self):
        if self._shuffle:
            rng = np.random.RandomState(self._random_seed)
            self._sample_ids = rng.permutation(self._sample_ids)

    def __len__(self) -> int:
        return math.ceil(self._num_samples / self._batch_size)

    def __iter__(self) -> Iterator[HSTUBatch]:
        max_total = self._max_history_seqlen + self._max_num_candidates

        for batch_idx in range(len(self)):
            start = batch_idx * self._batch_size
            end = min(start + self._batch_size, self._num_samples)
            ids = self._sample_ids[start:end]
            actual_bs = end - start

            contextual_values: Dict[str, List[int]] = {n: [] for n in self.CONTEXTUAL_FEATURES}
            item_values: List[int] = []
            item_lengths: List[int] = []
            action_values: List[int] = []
            action_lengths: List[int] = []
            num_candidates: List[int] = []
            labels: List[int] = []

            for sid in ids:
                row = self._df.iloc[sid]

                for cf in self.CONTEXTUAL_FEATURES:
                    contextual_values[cf].append(int(row[cf]))

                raw_items = maybe_truncate_seq(load_seq(row[self.ITEM_FEATURE]), max_total)
                raw_actions = maybe_truncate_seq(load_seq(row[self.ACTION_FEATURE]), max_total)

                item_values.extend(raw_items)
                item_lengths.append(len(raw_items))

                history_actions = raw_actions[: -self._max_num_candidates]
                candidate_actions = raw_actions[-self._max_num_candidates :]
                action_seq = history_actions + candidate_actions
                action_values.extend(action_seq)
                action_lengths.append(len(action_seq))

                if self._max_num_candidates > 0:
                    nc = min(self._max_num_candidates, len(raw_items))
                    num_candidates.append(nc)

                if self._num_tasks > 0:
                    label = raw_actions[-self._max_num_candidates :] if self._max_num_candidates > 0 else action_seq
                    labels.extend(label)

            pad_n = self._batch_size - actual_bs

            def pad_tensor(t, pad_n):
                return torch.nn.functional.pad(t, (0, pad_n)) if pad_n > 0 else t

            ctx_tensor = torch.tensor(
                [contextual_values[n] for n in self.CONTEXTUAL_FEATURES]
            ).view(-1)
            ctx_lengths = pad_tensor(
                torch.ones(actual_bs, dtype=torch.long), pad_n
            )

            item_values_t = torch.tensor(item_values, dtype=torch.long)
            item_lengths_t = pad_tensor(torch.tensor(item_lengths, dtype=torch.long), pad_n)

            action_values_t = torch.tensor(action_values, dtype=torch.long)
            action_lengths_t = pad_tensor(torch.tensor(action_lengths, dtype=torch.long), pad_n)

            num_candidates_t = pad_tensor(torch.tensor(num_candidates, dtype=torch.long), pad_n)

            all_keys = self.CONTEXTUAL_FEATURES + [self.ITEM_FEATURE, self.ACTION_FEATURE]
            all_values = torch.cat([ctx_tensor, item_values_t, action_values_t])
            all_lengths = torch.cat([
                ctx_lengths.repeat(len(self.CONTEXTUAL_FEATURES)),
                item_lengths_t,
                action_lengths_t,
            ])

            features = KeyedJaggedTensor(
                keys=all_keys,
                values=all_values,
                lengths=all_lengths,
            )

            label_jt = None
            if self._num_tasks > 0:
                label_lengths = (
                    num_candidates_t if self._max_num_candidates > 0
                    else item_lengths_t.clone()
                )
                label_jt = JaggedTensor(
                    values=torch.tensor(labels, dtype=torch.long),
                    lengths=label_lengths,
                )

            yield HSTUBatch(
                features=features,
                labels=label_jt,
                batch_size=self._batch_size,
                item_feature_name=self.ITEM_FEATURE,
                action_feature_name=self.ACTION_FEATURE,
                contextual_feature_names=self.CONTEXTUAL_FEATURES,
                feature_to_max_seqlen=self._feature_to_max_seqlen,
                max_num_candidates=self._max_num_candidates,
                num_candidates=num_candidates_t if self._max_num_candidates > 0 else None,
            )
