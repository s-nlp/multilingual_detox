import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    MT5ForConditionalGeneration,
    MT5Tokenizer,
    Trainer,
    TrainingArguments,
)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_only_russian(part="train"):
    rus_data = pd.read_csv(f"data/russian_data/{part}.tsv", sep="\t")
    rus_data = rus_data[["toxic_comment", "neutral_comment"]]
    rus_data["prefix"] = ["ru_RU" for _ in range(rus_data.shape[0])]

    return rus_data


def load_data(
    set_iter: int = 7,
    drop_duplicates: bool = False,
    use_russian: bool = False,
    part: float = 1.0,
) -> pd.DataFrame:
    data = pd.read_excel(
        f"results_second_launch/aggregated/for_experiments/jigsaw_twitter_reddit_{set_iter}.xlsx"
    )
    data = data[data.neutral_comment.apply(lambda x: isinstance(x, str)) == True]
    data.reset_index(inplace=True)
    for col in ["idx", "input_task1_suite_id", "index"]:
        try:
            data.drop(columns=[col], axis=1, inplace=True)
        except KeyError:
            continue
    data.dropna(inplace=True)

    if drop_duplicates:
        data.drop_duplicates(subset="toxic_comment", keep="last", inplace=True)
    data["prefix"] = ["en_XX" for _ in range(data.shape[0])]

    if use_russian:
        rus_data = pd.read_csv("data/russian_data/train.tsv", sep="\t")
        rus_data = rus_data[["toxic_comment", "neutral_comment"]]
        rus_data["prefix"] = ["ru_RU" for _ in range(rus_data.shape[0])]
        data = pd.concat(
            [
                data[["prefix", "toxic_comment", "neutral_comment"]].iloc[
                    : rus_data.shape[0]
                ],
                rus_data.iloc[: int(rus_data.shape[0] * part)],
            ]
        )
    return data


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
    if (
        tokenizer.sep_token_id in (None, tokenizer.vocab_size)
        and "bos_token" in special_tokens
    ):
        special_tokens["sep_token"] = special_tokens["bos_token"]
    tokenizer.add_special_tokens(special_tokens)

    print("Vocab size: ", tokenizer.vocab_size)
    print("PAD: ", tokenizer.pad_token_id, tokenizer.pad_token)
    print("BOS: ", tokenizer.bos_token_id, tokenizer.bos_token)
    print("EOS: ", tokenizer.eos_token_id, tokenizer.eos_token)
    print("UNK: ", tokenizer.unk_token_id, tokenizer.unk_token)
    print("SEP: ", tokenizer.sep_token_id, tokenizer.sep_token)
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size", type=int, default=8, help="batch size (8 or 16)"
    )
    parser.add_argument(
        "--use_russian", type=int, default=1, help="flag whether to use russian or not"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=5_000,
        help="maximum training steps (1000, 3000, 5000 or 10000)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-5, help="weight decay better be 1e-5"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-5, help="learning rate"
    )
    parser.add_argument(
        "--output_dir", type=str, default="mt5s", help="name for trained models folder"
    )
    parser.add_argument("--n_device", type=int, default="0")
    args = parser.parse_args()

    set_random_seed(1337)

    os.environ["WANDB_DISABLED"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.n_device}"

    data = load_data(use_russian=bool(args.use_russian))
    # train_data = load_only_russian(part='train')
    # tune_data = load_only_russian(part='dev')

    train_ids, tune_ids = train_test_split(
        data.index, test_size=0.01, random_state=1337
    )

    model = MT5ForConditionalGeneration.from_pretrained("google/mt5-base")
    tokenizer = MT5Tokenizer.from_pretrained("google/mt5-base")
    tokenizer = fix_tokenizer(tokenizer)
    model.resize_token_embeddings(len(tokenizer))
    device = torch.device("cuda")
    model = model.to(device)

    trainset = ToxicDataset(data.iloc[train_ids], tokenizer)
    tuneset = ToxicDataset(data.iloc[tune_ids], tokenizer)

    if args.use_russian == 1:
        flag = "EN_RU"
    else:
        flag = "RU"
    args = TrainingArguments(
        output_dir=f"{args.output_dir}/mt5_{args.max_steps}_{flag}",
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="steps",
        weight_decay=args.weight_decay,
        logging_steps=100,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        seed=1337,
        save_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=trainset,
        eval_dataset=tuneset,
        tokenizer=tokenizer,
    )
    trainer.train()
