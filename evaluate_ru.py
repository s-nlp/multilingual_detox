import os

import numpy as np
import pandas as pd
import torch
from tqdm.auto import trange
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from argparse import ArgumentParser


def prepare_target_label(model, target_label):
    if target_label in model.config.id2label:
        pass
    elif target_label in model.config.label2id:
        target_label = model.config.label2id.get(target_label)
    elif target_label.isnumeric() and int(target_label) in model.config.id2label:
        target_label = int(target_label)
    else:
        raise ValueError(
            f'target_label "{target_label}" not in model labels or ids: {model.config.id2label}.'
        )
    return target_label


def classify_texts(
    model,
    tokenizer,
    texts,
    second_texts=None,
    target_label=None,
    batch_size=32,
    verbose=False,
):
    target_label = prepare_target_label(model, target_label)
    res = []
    if verbose:
        tq = trange
    else:
        tq = range
    for i in tq(0, len(texts), batch_size):
        inputs = [texts[i : i + batch_size]]
        if second_texts is not None:
            inputs.append(second_texts[i : i + batch_size])
        inputs = tokenizer(
            *inputs, return_tensors="pt", padding=True, truncation=True, max_length=512,
        ).to(model.device)
        with torch.no_grad():
            try:
                preds = (
                    torch.softmax(model(**inputs).logits, -1)[:, target_label]
                    .cpu()
                    .numpy()
                )
            except:
                print(i, i + batch_size)
                preds = [0] * len(inputs)
        res.append(preds)
    return np.concatenate(res)


def evaluate_style(
    model,
    tokenizer,
    texts,
    target_label=1,  # 1 is formal, 0 is informal
    batch_size=32,
    verbose=False,
):
    target_label = prepare_target_label(model, target_label)
    scores = classify_texts(
        model,
        tokenizer,
        texts,
        batch_size=batch_size,
        verbose=verbose,
        target_label=target_label,
    )
    return scores


def evaluate_meaning(
    model,
    tokenizer,
    original_texts,
    rewritten_texts,
    target_label="entailment",
    bidirectional=True,
    batch_size=32,
    verbose=False,
    aggregation="prod",
):
    target_label = prepare_target_label(model, target_label)
    scores = classify_texts(
        model,
        tokenizer,
        original_texts,
        rewritten_texts,
        batch_size=batch_size,
        verbose=verbose,
        target_label=target_label,
    )
    if bidirectional:
        reverse_scores = classify_texts(
            model,
            tokenizer,
            rewritten_texts,
            original_texts,
            batch_size=batch_size,
            verbose=verbose,
            target_label=target_label,
        )
        if aggregation == "prod":
            scores = reverse_scores * scores
        elif aggregation == "mean":
            scores = (reverse_scores + scores) / 2
        elif aggregation == "f1":
            scores = 2 * reverse_scores * scores / (reverse_scores + scores)
        else:
            raise ValueError('aggregation should be one of "mean", "prod", "f1"')
    return scores


def encode_cls(texts, model, tokenizer, batch_size=32, verbose=False):
    results = []
    if verbose:
        tq = trange
    else:
        tq = range
    for i in tq(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        with torch.no_grad():
            out = model(
                **tokenizer(
                    batch, return_tensors="pt", padding=True, truncation=True
                ).to(model.device)
            )
            embeddings = out.pooler_output
            embeddings = torch.nn.functional.normalize(embeddings).cpu().numpy()
            results.append(embeddings)
    return np.concatenate(results)


def evaluate_cosine_similarity(
    model, tokenizer, original_texts, rewritten_texts, batch_size=32, verbose=False,
):
    scores = (
        encode_cls(
            original_texts,
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            verbose=verbose,
        )
        * encode_cls(
            rewritten_texts,
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            verbose=verbose,
        )
    ).sum(1)
    return scores


def evaluate_cola(model, tokenizer, texts, target_label=1, batch_size=32, verbose=False):
    target_label = prepare_target_label(model, target_label)
    scores = classify_texts(
        model,
        tokenizer,
        texts,
        batch_size=batch_size,
        verbose=verbose,
        target_label=target_label,
    )
    return scores


def evaluate_cola_relative(
    model,
    tokenizer,
    original_texts,
    rewritten_texts,
    target_label=1,
    batch_size=32,
    verbose=False,
    maximum=0,
):
    target_label = prepare_target_label(model, target_label)
    original_scores = classify_texts(
        model,
        tokenizer,
        original_texts,
        batch_size=batch_size,
        verbose=verbose,
        target_label=target_label,
    )
    rewritten_scores = classify_texts(
        model,
        tokenizer,
        rewritten_texts,
        batch_size=batch_size,
        verbose=verbose,
        target_label=target_label,
    )
    scores = rewritten_scores - original_scores
    if maximum is not None:
        scores = np.minimum(0, scores)
    return scores


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def rotation_calibration(data, coef=1.0, px=1, py=1, minimum=0, maximum=1):
    result = (data - px) * coef + py
    if minimum is not None:
        result = np.maximum(minimum, result)
    if maximum is not None:
        result = np.minimum(maximum, result)
    return result


def evaluate_style_transfer(
    original_texts,
    rewritten_texts,
    style_model,
    style_tokenizer,
    meaning_model,
    meaning_tokenizer,
    cola_model,
    cola_tokenizer,
    style_target_label=1,
    batch_size=32,
    verbose=True,
    aggregate=False,
    style_calibration=None,
    meaning_calibration=None,
    fluency_calibration=None,
):
    if verbose:
        print("Style evaluation")
    accuracy = evaluate_style(
        style_model,
        style_tokenizer,
        rewritten_texts,
        target_label=style_target_label,
        batch_size=batch_size,
        verbose=verbose,
    )
    if verbose:
        print("Meaning evaluation")
    similarity = evaluate_cosine_similarity(
        meaning_model,
        meaning_tokenizer,
        original_texts,
        rewritten_texts,
        batch_size=batch_size,
        verbose=verbose,
    )
    if verbose:
        print("Fluency evaluation")
    fluency = evaluate_cola_relative(
        cola_model,
        cola_tokenizer,
        rewritten_texts=rewritten_texts,
        original_texts=original_texts,
        batch_size=batch_size,
        verbose=verbose,
    )

    joint = accuracy * similarity * fluency
    if verbose:
        print(f"Style accuracy:       {np.mean(accuracy)}")
        print(f"Meaning preservation: {np.mean(similarity)}")
        print(f"Joint fluency:        {np.mean(fluency)}")
        print(f"Joint score:          {np.mean(joint)}")

    # Calibration
    if style_calibration:
        accuracy = style_calibration(accuracy)
    if meaning_calibration:
        similarity = meaning_calibration(similarity)
    if fluency_calibration:
        fluency = fluency_calibration(fluency)
    joint = accuracy * similarity * fluency
    if verbose and (style_calibration or meaning_calibration or fluency_calibration):
        print("Scores after calibration:")
        print(f"Style accuracy:       {np.mean(accuracy)}")
        print(f"Meaning preservation: {np.mean(similarity)}")
        print(f"Joint fluency:        {np.mean(fluency)}")
        print(f"Joint score:          {np.mean(joint)}")

    result = dict(accuracy=accuracy, similarity=similarity, fluency=fluency, joint=joint)
    if aggregate:
        return {k: float(np.mean(v)) for k, v in result.items()}
    return result


def evaluate(original, rewritten):
    return evaluate_style_transfer(
        original_texts=original,
        rewritten_texts=rewritten,
        style_model=style_model,
        style_tokenizer=style_tokenizer,
        meaning_model=meaning_model,
        meaning_tokenizer=meaning_tokenizer,
        cola_model=cola_model,
        cola_tokenizer=cola_tolenizer,
        style_target_label=0,
        aggregate=True,
        style_calibration=lambda x: rotation_calibration(x, 0.90),
        meaning_calibration=lambda x: rotation_calibration(x, 1.50),
        fluency_calibration=lambda x: rotation_calibration(x, 1.15, px=0),
    )


def load_model(
    model_name=None,
    model=None,
    tokenizer=None,
    model_class=AutoModelForSequenceClassification,
    use_cuda=True,
):
    if model is None:
        if model_name is None:
            raise ValueError("Either model or model_name should be provided")
        model = model_class.from_pretrained(model_name)
        if torch.cuda.is_available() and use_cuda:
            model.cuda()
    if tokenizer is None:
        if model_name is None:
            raise ValueError("Either tokenizer or model_name should be provided")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--result_filename",
        type=str,
        default="results_ru",
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
    args = parser.parse_args()

    style_model, style_tokenizer = load_model(
        "SkolkovoInstitute/russian_toxicity_classifier", use_cuda=True
    )
    meaning_model, meaning_tokenizer = load_model(
        "cointegrated/LaBSE-en-ru", use_cuda=True, model_class=AutoModel
    )
    cola_model, cola_tolenizer = load_model(
        "SkolkovoInstitute/rubert-base-corruption-detector", use_cuda=True
    )

    inputs = pd.read_csv("data/russian_data/test.tsv", sep="\t")["toxic_comment"].values

    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(f"{args.output_dir}/{args.result_filename}.md"):
        with open(f"{args.output_dir}/{args.result_filename}.md", "w") as file:
            file.write(f"| Model name | STA | SIM | FL | J |\n")

    results = []
    with open(f"{args.input_dir}/results_ru.txt", "r") as file:
        preds = file.readlines()
        preds = [sentence.strip() for sentence in preds]

        assert all(len(x) > 0 for x in inputs)
        assert all(len(x) > 0 for x in preds)
        assert all(isinstance(x, str) for x in inputs)
        assert all(isinstance(x, str) for x in preds)

        result = evaluate(inputs.tolist(), preds)
        r = f"{args.input_dir}|{result['accuracy']:.3f}|{result['similarity']:.3f}|{result['fluency']:.3f}|{result['joint']:.3f}\n"

        with open(f"{args.output_dir}/{args.result_filename}.md", "a") as file:
            file.write(r)

