What's In A Name? Evaluating Assembly-Part Semantic Knowledge in Language Models through User-Provided Names in CAD Files
===============

Contents:

1. [Citing this Work](#Citing-this-Work)
2. [Data](#Data)
3. [Directory Structure](#Directory-Structure)
4. [Requirements](#Requirements)
5. [Data](#Data)
6. [Code](#Code)
   1. [Structure](#Structure)
   2. [Running the Models](#Running-the-Models)
      1. [Fine-Tuning DistilBERT](#Fine-Tuning-DistilBERT)
      2. [Two Parts](#Two-Parts)
      3. [Missing Part & Document Name](#Missing-Part-&-Document-Name)
7. [License](#License)

### Citing this Work

The full paper is available on arXiv at: <https://arxiv.org/abs/2304.14275>

If you use any of the code or techniques from this repository, please cite the following:

> Peter Meltzer, Joseph G. Lambourne and Daniele Grandi. ‘What's In A Name? Evaluating Assembly-Part Semantic Knowledge in Language Models through User-Provided Names in CAD Files’. In Journal of Computing and Information Science in Engineering (JCISE), 2023. https://arxiv.org/abs/2304.14275

### Data

Data is available from <https://whats-in-a-name-dataset.s3.us-west-2.amazonaws.com/whats_in_a_name_dataset.zip>.

This can be downloaded into place using:

```shell
$ ./download_data.sh
```

(requires curl and unzip).

### Directory Structure

Ensure the data and source code are in the same root directory, i.e.:

```
whats_in_a_name_code_data
├── README.md
├── data
│   └── abc
│       ├── abc_text_data_003.json
│       │   ...
│       └── validation_pairs.csv
├── requirements.txt
└── src
    ├── abcpartnames
    │   ├── __init__.py
    │   ├── data_processing
    │   │   ├── __init__.py
    │   │   │ ...
    │   │   └── generate_pairs.py
    │   │   ...
    │   └── transforms
    │       ├── __init__.py
    │       │ ...
    │       └── vectorizers.py
    └── setup.py
```

### Requirements

- python 3.9

We recommend using a virtual environment, then:

```shell
pip install -r requirements.txt
```

### Data

`abc_text_data_003.json` is the complete set of data extracted with default part names removed. It contains a
dictionary with an entry for each OnShape document in ABC dataset. This is deduplicated based on part names and feature 
names. Where names or descriptions are unavailable empty lists and empty strings are used.

```python
{
  "OnShape_document_id": {
    "body_names": List[str],
    "feature_names": List[str],
    "document_name": str,
    "document_description": str
  },
  ...
}
```

`validation_pairs.csv`, `test_pairs.csv` are the generated positive and negative pairs for "Two Parts" experiment.

`label_1...label_N` are 1 for a positive pair (two parts from same document) or 0 for a negative pair (parts are not 
from same document). `part_a_1...part_a_N` and `part_b_1...part_b_N` are the part name strings. All positive pairs are 
first (i.e. rows 1...k) and negative pairs last (i.e. rows k+1...N).

```csv
label_1,part_a_1,part_b_1
...
label_N,part_a_N,part_b_N
```

`train_corpus_complete.txt` is the training corpus generated (from the train set) for fine-tuning DistilBERT or fully
training FastText.

```text
“An assembly with name {DOCUMENT NAME} contains the following parts: {PART 1}, ..., {PART N}.\n”
...
```

`train_val_test_split.json` is the master lists of splits for this dataset. Each train/validation/test entry contains
a list of OnShape document ids for the documents corresponding to that split.

```python
{
  "train": List[str],
  "validation": List[str],
  "test": List[str]
}
```

The following files are derived from the splits defined in `train_val_test_split.json` and follow the same format:

- `train_val_test_descriptions.json` - documents having a non-empty description
- `train_val_test_featurenames.json` - documents with at least 1 non-default feature name
- `train_val_test_partnames.json` - documents with at least 1 non-default part name
- `train_val_test_partnames_and_featurenames.json` - documents with at least 1 non-default part and feature name
- `train_val_test_two_or_more_partnames.json` - documents with at least 2 non-default part names

Not all of these files were used in our experiments, but we share them for future use.

### Code

#### Structure

`src/abcpartnames/`:

- `data_processing/`
  - `create_train_val_test_split.py`: create train, validation and test splits
  - `generate_corpus.py`: create fine-tuning corpus (also used for training FastText)
  - `generate_pairs.py`: generate positive and negative pairs from document parts
- `datasets/`
  - `ABCTextDataset.py`: all `Dataset` classes and `DataLoaders` etc.
- `models/`
  - `mlp.py`: MLP used in "Two Parts" experiment
  - `set_model.py`: end-to-end module (`SetModuleNamesWithParts`) that combines the encoder with the SetTransformer
  - `set_model.py`: implementation of SetTransformer model used in "Missing Part" and "Document Name" experiments
- `scripts/`
  - `train_vectorizers.py`, `evaluate_vectorizers.py`: train/evaluate the BOW vectorizers on "Two Parts" experiment
  - `evaluate_technet.py`: evaluate TechNet on "Two Parts" experiment
  - `train_word2vec.py`, `evaluate_fasttext_models.py`: train/evaluate FastText on "Two Parts" experiment
  - `evaluate_embeddings.py`: evaluate DistilBERT(-FT) on "Two Parts" experiment
  - `train_and_eval_set_model.py`: evaluate any of TechNet, FastText, DistilBERT and DistilBERT-FT on "Missing Part" or "Document Name" experiments
  - `run_mlm.py`: (script from huggingface) fine-tune DistilBERT 
- `transforms/`
  - `hf.py`: transforms for huggingface models (DistilBERT + DistilBERT-FT), includes pooling
  - `lower_and_replace_transform.py`: transform to convert strings to lowercase and replace `_` with ` `
  - `vectorizers.py`: BOW Vectorizers, FastText encoder and TechNet encoder

#### Running the Models

For all scripts, run from the project root directory and use `-h` flag to see possible arguments and default options.


##### Fine-Tuning DistilBERT

```shell
$ python src/abcpartnames/scripts/run_mlm.py \
    --model_name_or_path distilbert-base-uncased \
    --line_by_line \
    --train_file data/abc/train_corpus_complete.txt \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --output_dir language_models/distilbert_fine-tuned_003
```

##### Two Parts

For training the vectorizers (TF-IDF BOW and Frequency BOW) use `$ python src/abcpartnames/scripts/train_vectorizers.py`,
and after for evaluation use `$ python src/abcpartsnames/scripts/evaluate_vectorizers.py`.

For TechNet use `$ python src/abcpartnames/scripts/evaluate_technet.py`.

For FastText training use `$ python src/abcpartnames/scripts/train_word2vec.py`, and for evaluation use `$ python src/abcpartnames/scripts/evaluate_fasttext_models.py`.

For DistilBERT and DistilBERT-FT evaluation use `$ python src/abcpartnames/scripts/evaluate_embeddings.py`
(passing path to root dir of fine-tuned model as `--model` for DistilBERT-FT). For example,

```shell
# evaluate DistilBERT on "Two Parts" experiment
$ python src/abcpartnames/scripts/evaluate_embeddings.py \
    --batch_size 512 \
    --num_workers 8 \   # set to 0 for debugging
    --accelerator gpu \
    --gpus 1 \
    --trial 0 \
    --exp pairs_distilbert \    # this will be used for saving lightning_logs
    --model distilbert-base-uncased   # this is the pre-trained model (it will be downloaded if it is not already)
```

```shell
# evaluate DistilBERT-FT on "Two Parts" experiment
$ python src/abcpartnames/scripts/evaluate_embeddings.py \
    --batch_size 512 \
    --num_workers 8 \   # set to 0 for debugging
    --accelerator gpu \
    --gpus 1 \
    --trial 0 \
    --exp pairs_distilbert-ft \    # this will be used for saving lightning_logs
    --model ./language_models/distilbert_fine-tuned_003   # this is the fine-tuned model (assumes model is already saved to this path as above)
```

##### Missing Part & Document Name

All experiments can be run using the script `src/abc/partnames/scripts/train_and_eval_set_model.py`.

Examples:

```shell
# train and evaluate pre-trained DistilBERT on "Missing Part" experiment
$ python src/abcpartnames/scripts/train_and_eval_set_model.py \
    --batch_size 512 \
    --trial 0 \
    --exp missing-part_distilbert \
    --model distilbert-base-uncased \
    --dim_hidden 768 \
    --num_heads 8 \
    --num_inds 64
```

```shell
# train and evaluate fine-tuned DistilBERT-FT on "Missing Part" experiment
$ python src/abcpartnames/scripts/train_and_eval_set_model.py \
    --batch_size 512 \
    --trial 0 \
    --exp missing-part_distilbert-ft \
    --model ./language_models/distilbert_fine-tuned_003 \
    --dim_hidden 768 \
    --num_heads 8 \
    --num_inds 64
```

```shell
# train and evaluate pre-trained DistilBERT on "Document Name" experiment
$ python src/abcpartnames/scripts/train_and_eval_set_model.py \
    --batch_size 512 \
    --trial 0 \
    --exp document-name_distilbert \
    --model distilbert-base-uncased \
    --dim_hidden 512 \
    --num_heads 8 \
    --num_inds 64 \
    --pred_names True
```

```shell
# train and evaluate fine-tuned DistilBERT-FT on "Document Name" experiment
$ python src/abcpartnames/scripts/train_and_eval_set_model.py \
    --batch_size 512 \
    --trial 0 \
    --exp document-name_distilbert-ft \
    --model ./language_models/distilbert_fine-tuned_003 \
    --dim_hidden 512 \
    --num_heads 8 \
    --num_inds 64 \
    --pred_names True
```

For all other models and options see below:

```
$ python src/abcpartnames/scripts/train_and_eval_set_model.py -h
usage: train_and_eval_set_model.py [-h] [--trial TRIAL] [--model MODEL] [--exp EXP] [--results RESULTS] [--workers WORKERS] [--stop_on STOP_ON] [--num_inds NUM_INDS] [--dim_input DIM_INPUT]
                                   [--dim_hidden DIM_HIDDEN] [--dim_out DIM_OUT] [--num_heads NUM_HEADS] [--ln] [--batch_size BATCH_SIZE] [--bin_width BIN_WIDTH] [--names NAMES]
                                   [--gensim_path GENSIM_PATH] [--pred_names PRED_NAMES] [--no_lower] [--replace_]

optional arguments:
  -h, --help            show this help message and exit
  --trial TRIAL         trial (used for random seed) (default: 0)
  --model MODEL         pre-trained language model (default: distilbert-base-uncased)   # or 'technet', 'fasttext', or path to fine-tuned distilbert
  --exp EXP             experiment_name (default: test)
  --results RESULTS     file to save results (default: set_results.csv)
  --workers WORKERS     num workers for dataloaders (default: 8)
  --stop_on STOP_ON     'loss' or 'acc' (default: loss)

SetModule:
  --num_inds NUM_INDS   number inducing points (default: 32)
  --dim_input DIM_INPUT
                        input dimensions to set transformer (default: 768)
  --dim_hidden DIM_HIDDEN
                        set transformer hidden units size (default: 512)
  --dim_out DIM_OUT     set transformer output dimensions (default: 768)
  --num_heads NUM_HEADS
                        number of attention heads (default: 4)
  --ln                  use layer norm (default: False)
  --batch_size BATCH_SIZE
                        batch size (default: 512)
  --bin_width BIN_WIDTH
                        width of bins to use in batched masking (default: 128)
  --names NAMES         include assembly names with input parts (default: False)

ToVectorized:
  --gensim_path GENSIM_PATH
                        path to saved FastText model (default: None)

ABCNamesWithParts:
  --pred_names PRED_NAMES
                        predict assembly names using all parts as input (default: False)
  --no_lower            do not convert strings to lower case (default: False)
  --replace_            Replace _ characters with spaces (default: False)
```
        
### License
Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg