import argparse
import os

import torch
from sklearn.model_selection import train_test_split
from transformers import (MT5ForConditionalGeneration, MT5Tokenizer, Trainer,
                          TrainingArguments)

from utils import (ToxicDataset, fix_tokenizer, load_data, load_only_russian,
                   set_random_seed)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size", type=int, default=8, help="batch size (8 or 16)"
    )
    parser.add_argument(
        "--use_russian",
        type=int,
        default=1,
        help="flag whether to use russian or not",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=5_000,
        help="maximum training steps (1000, 3000, 5000 or 10000)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="weight decay better be 1e-5",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-5, help="learning rate"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="mt5s",
        help="name for trained models folder",
    )
    parser.add_argument("--n_device", type=int, default="0")
    parser.add_argument("--seed", type=int, required=False, default=42)
    args = parser.parse_args()

    set_random_seed(args.seed)

    os.environ["WANDB_DISABLED"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.n_device}"

    train_data, tune_data = load_data(use_russian=bool(args.use_russian))
    # train_data = load_only_russian(part='train')
    # tune_data = load_only_russian(part='dev')

#     train_ids, tune_ids = train_test_split(
#         data.index, test_size=0.01, random_state=args.seed
#     )

    model = MT5ForConditionalGeneration.from_pretrained("google/mt5-base")
    tokenizer = MT5Tokenizer.from_pretrained("google/mt5-base")
    tokenizer = fix_tokenizer(tokenizer)
    model.resize_token_embeddings(len(tokenizer))
    device = torch.device("cuda")
    model = model.to(device)

    trainset = ToxicDataset(train_data, tokenizer)
    tuneset = ToxicDataset(tune_data, tokenizer)

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
        seed=args.seed,
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
    print("training started")
    model.save_pretrained(f"{args.output_dir}/mt5_{args.max_steps}_{flag}")
    tokenizer.save_pretrained(f"{args.output_dir}/mt5_{args.max_steps}_{flag}")
