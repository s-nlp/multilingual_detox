## Exploring Cross-lingual Textual Style Transfer with Large Multilingual Language Models
This repository contains code for the paper submitted to [ACL SRW](https://sites.google.com/view/acl-srw-2022/home) **Exploring Cross-lingual Textual Style Transfer with Large Multilingual Language Models**
by [Daniil Moskovskiy](mailto:daniil.moskovskiy@skoltech.ru), [Daryna Dementieva](mailto:daryna.dementieva@skoltech.ru) and [Alexander Panchenko](mailto:a.panchenko@skoltech.ru)

## Setup

**Step $1$: Install dependencies**

```
pip install -r requirements.txt
```

**Step $2$: Run Experiments**

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
    --use_russian 0 \
    --max_steps 40000 \
    --learning_rate 1e-5 \
    --output_dir trained_models 
```

**Step $3$: Generate detoxifications for test data**

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


**Step $4$**

Calculate metrics. Example for Russian:

```
python evaluate_ru.py \
    --result_filename results_en \
    --input_dir mbarts/mbart_10000_EN_RU \
    --output_dir mbarts \
```


## Data

For English we use ParaDetox parallel detoxification corpora, please, cite the original [paper](https://aclanthology.org/2022.acl-long.469/) and proceed to the original ParaDetox [repository](https://github.com/skoltech-nlp/paradetox) for details. For Russian we use RuDetox corpora from [RuSSE Detoxification Competition](https://russe.nlpub.org/2022/tox/), please cite the competition if you are going to use the data.

Citation for English data:
```
@inproceedings{logacheva-etal-2022-paradetox,
    title = "{P}ara{D}etox: Detoxification with Parallel Data",
    author = "Logacheva, Varvara  and
      Dementieva, Daryna  and
      Ustyantsev, Sergey  and
      Moskovskiy, Daniil  and
      Dale, David  and
      Krotova, Irina  and
      Semenov, Nikita  and
      Panchenko, Alexander",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.469",
    pages = "6804--6818",
    abstract = "We present a novel pipeline for the collection of parallel data for the detoxification task. We collect non-toxic paraphrases for over 10,000 English toxic sentences. We also show that this pipeline can be used to distill a large existing corpus of paraphrases to get toxic-neutral sentence pairs. We release two parallel corpora which can be used for the training of detoxification models. To the best of our knowledge, these are the first parallel datasets for this task.We describe our pipeline in detail to make it fast to set up for a new language or domain, thus contributing to faster and easier development of new parallel resources.We train several detoxification models on the collected data and compare them with several baselines and state-of-the-art unsupervised approaches. We conduct both automatic and manual evaluations. All models trained on parallel data outperform the state-of-the-art unsupervised models by a large margin. This suggests that our novel datasets can boost the performance of detoxification systems.",
}
```

Citation for Russian data:

```
@article{russe2022detoxification,
  title={RUSSE-2022: Findings of the First Russian Detoxification Task Based on Parallel Corpora},
  author={Dementieva, Daryna and Nikishina, Irina and Logacheva, Varvara and Fenogenova, Alena and Dale, David and Krotova, Irina and Semenov, Nikita and Shavrina, Tatiana and Panchenko, Alexander},
  booktitle={Computational Linguistics and Intellectual Technologies},
  year={2022}
}
```

## Results

Main results are depicted in this table below.

<p align="center">
  <img width="460" height="300" src="https://github.com/skoltech-nlp/multilingual_detox/blob/main/pics/main_table.png">
</p>

<!-- ![](https://github.com/skoltech-nlp/multilingual_detox/blob/main/pics/main_table.png) -->

Here are some examples of generated text.
<!-- ![](https://github.com/skoltech-nlp/multilingual_detox/blob/main/pics/generation.png) -->

<p align="center">
  <img width="460" height="300" src="https://github.com/skoltech-nlp/multilingual_detox/blob/main/pics/generation.png">
</p>

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
