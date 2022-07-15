from argparse import ArgumentParser

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    MT5ForConditionalGeneration,
    MT5Tokenizer,
    Trainer,
    TrainingArguments,
)

from utils import MT5Dataset, load_config, set_random_seed, load_cross_lin_data

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, default="configs/mt5_en_ru.json", required=True
    )
    arg = parser.parse_args()

    config = load_config(arg.config_path)

    model = MT5ForConditionalGeneration.from_pretrained(
        "google/mt5-{}".format(config["model_size"])
    )
    model = model.to(torch.device("cuda"))
    tokenizer = MT5Tokenizer.from_pretrained(
        "google/mt5-{}".format(config["model_size"])
    )

    set_random_seed(config["seed"])

    data = load_cross_lin_data(config["data_path"], direction=config["dir"])
    data.dropna(inplace=True)
    train_ids, valid_ids = train_test_split(
        data.index.values, test_size=0.05, random_state=config["seed"]
    )
    train_part = data.loc[train_ids]
    valid_part = data.loc[valid_ids]

    trainset = MT5Dataset(train_part, tokenizer)
    valset = MT5Dataset(valid_part, tokenizer)

    train_args = TrainingArguments(
        output_dir=config["output_dir"],
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=config["train_batch_size"],
        per_device_eval_batch_size=config["eval_batch_size"],
        save_strategy=config["save_strategy"],
        evaluation_strategy="steps",
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        logging_steps=config["logging_steps"],
        max_steps=config["max_steps"],
        learning_rate=config["learning_rate"],
        seed=config["seed"],
        save_steps=config["save_steps"],
        warmup_steps=config["warmup_steps"],
        run_name=config["model_name"],
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=trainset,
        eval_dataset=valset,
        tokenizer=tokenizer,
    )

    trainer.train()
