import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    MBart50TokenizerFast,
    MBartForConditionalGeneration,
    Trainer,
    TrainingArguments,
)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
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
):
    data = pd.read_excel(
        f"results_second_launch/aggregated/for_experiments/jigsaw_twitter_reddit_{set_iter}.xlsx"
    )
    data = data[
        data.neutral_comment.apply(lambda x: isinstance(x, str)) == True
    ]
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

    train_ids, tune_ids = train_test_split(
        data.index, test_size=0.01, random_state=1337
    )
    train_data = data.loc[train_ids]
    tune_data = data.loc[tune_ids]

    if use_russian:
        # train part
        rus_data_train = pd.read_csv("data/russian_data/train.tsv", sep="\t")
        rus_data_train = rus_data_train[["toxic_comment", "neutral_comment"]]
        rus_data_train["prefix"] = [
            "ru_RU" for _ in range(rus_data_train.shape[0])
        ]

        # tune part
        rus_data_tune = pd.read_csv("data/russian_data/dev.tsv", sep="\t")
        rus_data_tune = rus_data_tune[["toxic_comment", "neutral_comment"]]
        rus_data_tune["prefix"] = [
            "ru_RU" for _ in range(rus_data_tune.shape[0])
        ]
    train_data = pd.concat(
        [
            train_data[["prefix", "toxic_comment", "neutral_comment"]].iloc[
                : rus_data_train.shape[0]
            ],
            rus_data_train,
        ]
    )

    tune_data = pd.concat(
        [
            tune_data[["prefix", "toxic_comment", "neutral_comment"]],
            rus_data_tune,
        ]
    )

    return train_data, tune_data


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
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--use_russian",
        type=int,
        default=1,
        help="whether to use russian or not (0 or 1)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=5_000,
        help="maximum learning steps ([1000, 3000, 5000, 10000])",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="learning rate for fine-tuning ([1e-3, 1e-4, 1e-5, 3e-5])",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="number of warmup steps ([0, 100, 500, 1000])",
    )
    parser.add_argument(
        "--n_device",
        type=int,
        default=0,
        help="num of device (choose from 0 to 5)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="mbarts",
        help='set directory for model saving'
    )
    args = parser.parse_args()

    set_random_seed(1337)
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.n_device}"

    train_data, tune_data = load_data(use_russian=bool(args.use_russian))
    print("data loaded")

    model = MBartForConditionalGeneration.from_pretrained(
        "facebook/mbart-large-50"
    )
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))
    device = torch.device("cuda") if torch.cuda.is_available else "cpu"
    model = model.to(device)
    print("model loaded")
    trainset = ToxicDataset(train_data, tokenizer)
    tuneset = ToxicDataset(tune_data, tokenizer)

    if args.use_russian == 1:
        flag = "EN_RU"
    else:
        flag = "RU"
    train_args = TrainingArguments(
        output_dir=f"{args.output_dir}/mbart_{args.max_steps}_{flag}_{args.learning_rate}_{args.warmup_steps}",
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="steps",
        logging_steps=1_000,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        seed=1337,
        save_strategy="steps",
        save_stes=1_000,
        warmup_steps=args.warmup_steps,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=trainset,
        eval_dataset=tuneset,
        tokenizer=tokenizer
    )

    print("training started")
    trainer.train()
    model.save_pretrained(
        f"{args.output_dir}/mbart_{args.max_steps}_{flag}_{args.learning_rate}_{args.warmup_steps}"
    )
