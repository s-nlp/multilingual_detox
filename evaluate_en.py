import os
import pandas as pd
from argparse import ArgumentParser

from evaluate_ru import evaluate_style_transfer, load_model, run_evaluation


def evaluate(original, rewritten, references=None):
    return evaluate_style_transfer(
        original_texts=original,
        rewritten_texts=rewritten,
        references=references,
        style_model=style_model,
        style_tokenizer=style_tokenizer,
        meaning_model=meaning_model,
        meaning_tokenizer=meaning_tokenizer,
        cola_model=cola_model,
        cola_tokenizer=cola_tolenizer,
        style_target_label=0,
        meaning_target_label=0,
        cola_target_label=0,
        aggregate=True,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--result_filename",
        type=str,
        default="results_en",
        help="name of markdown file to save the results",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Directory with the generated .txt file",
        required=True,
    )
    parser.add_argument(
        "--output_dir", type=str, default="", help="Directory where to save the results",
    )
    parser.add_argument('-t', "--token", help="huggingface_token", default=None)
    args = parser.parse_args()

    style_model, style_tokenizer = load_model(
        "SkolkovoInstitute/roberta_toxicity_classifier", use_cuda=True, use_auth_token=args.token
    )
    meaning_model, meaning_tokenizer = load_model(
        "Elron/bleurt-large-128", use_cuda=True, use_auth_token=args.token
    )
    cola_model, cola_tolenizer = load_model(
        "cointegrated/roberta-large-cola-krishna2020", use_cuda=True, use_auth_token=args.token
    )

    with open('data/english_data/test_toxic_parallel.txt', 'r') as f:
        inputs = [line.strip() for line in f.readlines()]
    with open('data/english_data/test_references.txt', 'r') as f:
        refs = [line.strip() for line in f.readlines()]

    run_evaluation(args, inputs, refs, evaluator=evaluate, input_filename='results_en.txt')
