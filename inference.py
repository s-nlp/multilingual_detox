from argparse import ArgumentParser

import torch
from tqdm import tqdm
from transformers import (
    MBart50TokenizerFast,
    MBartForConditionalGeneration,
    MT5ForConditionalGeneration,
    MT5Tokenizer,
)
import pandas as pd


def paraphrase(
    text, model, tokenizer, n=None, max_length="auto", beams=5,
):
    texts = [text] if isinstance(text, str) else text
    inputs = tokenizer(texts, return_tensors="pt", padding=True)["input_ids"].to(
        model.device
    )

    if max_length == "auto":
        max_length = inputs.shape[1] + 10

    result = model.generate(
        inputs,
        num_return_sequences=n or 1,
        do_sample=False,
        temperature=1.0,
        repetition_penalty=10.0,
        max_length=max_length,
        min_length=int(0.5 * max_length),
        num_beams=beams,
        forced_bos_token_id=tokenizer.lang_code_to_id[tokenizer.tgt_lang],
    )
    texts = [tokenizer.decode(r, skip_special_tokens=True) for r in result]

    if not n and isinstance(text, str):
        return texts[0]
    return texts[0]


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        options=["mbart", "mt5"],
        help="Specify model type for loading",
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Specify path to saved model",
    )
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        options=["en", "ru"],
        help="Specify language for generation",
    )
    args = parser.parse_args()

    if args.language == "ru":
        test_data = pd.read_csv("data/russian_data/test.tsv", sep="\t")[
            "toxic_comment"
        ].values
        assert len(test_data) == 1000
    if args.language == "en":
        test_data = (
            open("data/english_data/test_toxic_parallel.txt", "r").read().split("\n")
        )
        assert len(test_data) == 671

    print(f"Loaded test {args.language} data")

    if args.model_name == "mbart":
        model = (
            MBartForConditionalGeneration.from_pretrained(f"{args.model_path}")
            .eval()
            .to(torch.device("cuda"))
        )
        tokenizer = MBart50TokenizerFast.from_pretrained(f"{args.model_path}")
    if args.model_name == "mt5":
        model = (
            MT5ForConditionalGeneration.from_pretrained(f"{args.model_path}")
            .eval()
            .to(torch.device("cuda"))
        )
        tokenizer = MT5Tokenizer.from_pretrained(f"{args.model_path}")

    print(f"Loaded {args.model_path} model")

    result = []
    for sentence in tqdm(test_data):
        out = paraphrase(sentence, model, tokenizer, n=1)
        result.append(out)

    with open(f"{args.model_path}/results_{args.language}.txt", "w") as f:
        f.write("\n".join(x for x in result))
