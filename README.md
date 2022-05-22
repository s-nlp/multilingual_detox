## Exploring Cross-lingual Textual Style Transfer with Large Multilingual Language Models
This repository contains code for the paper submitted to [ACL SRW](https://sites.google.com/view/acl-srw-2022/home) **Exploring Cross-lingual Textual Style Transfer with Large Multilingual Language Models**
by [Daniil Moskovskiy](mailto:daniil.moskovskiy@skoltech.ru), [Daryna Dementieva](mailto:daryna.dementieva@skoltech.ru) and [Alexander Panchenko](mailto:a.panchenko@skoltech.ru)

## Setup

**Step 1: Install dependencies**

```
pip install -r requirements.txt
```

**Step 2: Run Experiments**

Multilingual setup:

```
python mt5_trainer.py \
    --batch_size 16 \
    --use_russian 1 \
    --max_steps 40000 \
    --learning_rate 1e-5 \
    --output_dir trained_models 
```

Cross-lingual setup:

```
python mt5_trainer.py
    --batch_size 16 \
    --use_russian 1 \
    --max_steps 40000 \
    --learning_rate 1e-5 \
    --output_dir trained_models 
```

**Step 3: Generate detoxifications for test data**

For Russian use `test.tsv` file from `data\russian_data\`. For English use `data\english_data\test_toxic_parallel.txt`.

Example for Russian:
``` 
python inference.py \
    --model_name mbart \
    --model_path mbarts\mbart_10000_EN_RU \
    --language RU
```

and for English:

```
python inference.py \
    --model_name mbart \
    --model_path mbarts\mbart_10000_EN_RU \
    --language EN
```

## Citation

```
@inproceedings{DBLP:conf/acl/MoskovskiyDP22,
  author    = {Daniil Moskovskiy and
               Daryna Dementieva and
               Alexander Panchenko},
  editor    = {Samuel Louvan and
               Andrea Madotto and
               Brielen Madureira},
  title     = {Exploring Cross-lingual Text Detoxification with Large Multilingual
               Language Models},
  booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational
               Linguistics: Student Research Workshop, {ACL} 2022, Dublin, Ireland,
               May 22-27, 2022},
  pages     = {346--354},
  publisher = {Association for Computational Linguistics},
  year      = {2022},
  url       = {https://aclanthology.org/2022.acl-srw.26},
  timestamp = {Thu, 19 May 2022 16:52:59 +0200},
  biburl    = {https://dblp.org/rec/conf/acl/MoskovskiyDP22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## Contacts

If you find some issue, do not hesitate to add it to [Github Issues](https://github.com/skoltech-nlp/multilingual_detox/issues).

Feel free to contact Daniil Moskovskiy via [e-mail](mailto:daniil.moskovskiy@skoltech.ru) or [telegram](t.me/etomoscow)
