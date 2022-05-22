import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def fix_tokenizer(tokenizer):
    # Fixing broken tokenizers
    special_tokens = dict()
    for token_id in range(1000):
        token = tokenizer.convert_ids_to_tokens(token_id)
        if tokenizer.pad_token_id in (None, tokenizer.vocab_size) and "pad" in token:
            special_tokens["pad_token"] = token
        if tokenizer.bos_token_id in (None, tokenizer.vocab_size) and "<s>" in token:
            special_tokens["bos_token"] = token
        if tokenizer.eos_token_id in (None, tokenizer.vocab_size) and "</s>" in token:
            special_tokens["eos_token"] = token
        if tokenizer.unk_token_id in (None, tokenizer.vocab_size) and "unk" in token:
            special_tokens["unk_token"] = token
        if tokenizer.sep_token_id in (None, tokenizer.vocab_size) and "sep" in token:
            special_tokens["sep_token"] = token
    if (tokenizer.sep_token_id in (None, tokenizer.vocab_size) and "bos_token" in special_tokens
    ):
        special_tokens["sep_token"] = special_tokens["bos_token"]
    tokenizer.add_special_tokens(special_tokens)

    return tokenizer


class ToxicDataset(torch.utils.data.Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer, part: str = "both"):

        assert part in ["ru", "en", "both"]
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, idx):

        self.tokenizer.src_lang = self.data.iloc[idx].prefix
        self.tokenizer.tgt_lang = self.data.iloc[idx].prefix

        source = self.tokenizer(
            self.data.iloc[idx].toxic_comment,
            max_length=100,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer(
            self.data.iloc[idx].neutral_comment,
            max_length=100,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        source["labels"] = target["input_ids"]

        return {k: v.squeeze(0) for k, v in source.items()}

    def __len__(self):
        return self.data.shape[0]


def load_data(use_russian: bool = False):
    data = pd.read_excel("data/english_data/en_data.xlsx")
    data = data[data.neutral_comment.apply(lambda x: isinstance(x, str)) == True]
    data.reset_index(inplace=True)
    for col in ["idx", "input_task1_suite_id", "index"]:
        try:
            data.drop(columns=[col], axis=1, inplace=True)
        except KeyError:
            continue
    data.dropna(inplace=True)

    data["prefix"] = ["en_XX" for _ in range(data.shape[0])]

    train_ids, tune_ids = train_test_split(data.index, test_size=0.01, random_state=42)
    train_data = data.loc[train_ids]
    tune_data = data.loc[tune_ids]

    if use_russian:
        # train part
        rus_data_train = pd.read_csv("data/russian_data/train.tsv", sep="\t")
        rus_data_train = rus_data_train[["toxic_comment", "neutral_comment"]]
        rus_data_train["prefix"] = ["ru_RU" for _ in range(rus_data_train.shape[0])]

        # tune part
        rus_data_tune = pd.read_csv("data/russian_data/dev.tsv", sep="\t")
        rus_data_tune = rus_data_tune[["toxic_comment", "neutral_comment"]]
        rus_data_tune["prefix"] = ["ru_RU" for _ in range(rus_data_tune.shape[0])]
    train_data = pd.concat(
        [
            train_data[["prefix", "toxic_comment", "neutral_comment"]].iloc[
                : rus_data_train.shape[0]
            ],
            rus_data_train,
        ]
    )

    tune_data = pd.concat(
        [tune_data[["prefix", "toxic_comment", "neutral_comment"]], rus_data_tune]
    )

    return train_data, tune_data


def load_only_russian(part="train"):
    rus_data = pd.read_csv(f"data/russian_data/{part}.tsv", sep="\t")
    rus_data = rus_data[["toxic_comment", "neutral_comment"]]
    rus_data["prefix"] = ["ru_RU" for _ in range(rus_data.shape[0])]

    return rus_data
