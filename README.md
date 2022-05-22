# multilingual_detox
This repository contains code for the paper submitted to [ACL SRW](https://sites.google.com/view/acl-srw-2022/home) **Exploring Cross-lingual Textual Style Transfer with Large Multilingual Language Models**
by [Daniil Moskovskiy](mailto:daniil.moskovskiy@skoltech.ru), [Daryna Dementieva](mailto:daryna.dementieva@skoltech.ru) and[Alexander Panchenko](mailto:a.panchenko@skoltech.ru)

## Setup

**Step 1: Install dependencies**

```
pip install -r requirements.txt
```

**Step 2: Run Experiments**

Multilingual setup:

```
python mt5_trainer.py --batch_size 16 --use_russian 1 --max_steps 40000 --learning_rate 1e-5 --output_dir trained_models 
```

**Step 3: Generate detoxifications for test data**

For Russian use `test.tsv` file from `data\russian_data\`. For English use `data\english_data\test_toxic_parallel.txt`.
