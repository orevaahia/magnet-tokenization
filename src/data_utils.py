import os
import numpy as np
import torch
import random

from ast import literal_eval
from itertools import chain
from pathlib import Path
from typing import Optional, Dict

from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_dataset, interleave_datasets
from datasets import DatasetDict
from dataclasses import dataclass
from transformers import  AutoTokenizer



logger = get_logger(__name__)

BYTE_MODEL="google/byt5-small"
CACHE_DIR="cache"
SEED=42

np.set_printoptions(suppress=True)

def dinsert_special_token_ids(example, script_id):
    example["input_ids"].insert(0, script_id)
    return example


def dinsert_special_token(example, script_id ):
    """
    Insert script-id at the fronr of every sequence
    """
    example["input_ids"].insert(0, script_id)
    return example

def group_texts(examples, max_seq_length):
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < max_seq_length  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // max_seq_length) * max_seq_length

    result = {
                k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }

    return result


class MixtureByteVocab(object):
    """
    Create Byte Vocabulary
    """
    def __init__(self, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(BYTE_MODEL, extra_ids=0, cache_dir=CACHE_DIR, additional_special_tokens=kwargs["script_tokens"])
        print("Loaded tokenizer")
        self.script_to_id = kwargs["script_tokens"]

    @property
    def vocab_size(self):
        vocab_size = max(self.tokenizer.added_tokens_decoder.keys()) + 1
        return vocab_size


    def __len__(self):
        return self.vocab_size


class MagnetDataset(object):
    def __init__(self, file_paths: str,
                    seq_len: int,
                    accelerator: Accelerator,
                    language_to_script_id: Optional[Dict] = None,
                    *args,
                    **kwargs):

        self.seq_len=seq_len
        self.vocab = MixtureByteVocab(*args, **kwargs)
        self.language_to_script_id = language_to_script_id
        train_dict, validation_dict, test_dict = DatasetDict(), DatasetDict(), DatasetDict()

        for file_path in os.listdir(file_paths):
            language_folder = os.path.join(file_paths, file_path)
            data_files = {}
            for split in ["train", "validation", "test"]:
                data_files[split] = os.path.join(language_folder, f'{split}.txt')

            dataset = load_dataset(
                "text",
                data_files=data_files,
                cache_dir=CACHE_DIR,
                streaming=True)

            with accelerator.main_process_first():
                tokenized_datasets = dataset.map(self.tokenize_group_function, batched=True, remove_columns=["text"])

                tokenized_datasets = tokenized_datasets.map(
                        group_texts,
                        batched=True,
                        fn_kwargs={"max_seq_length": self.seq_len}
                    )
                # if routing via boundary predictors
                if language_to_script_id is not None:
                    tokenized_datasets = tokenized_datasets.map(dinsert_special_token,
                                        fn_kwargs={"script_id": self.language_to_script_id[file_path]})

                train_dict[file_path] = tokenized_datasets["train"]
                validation_dict[file_path] = tokenized_datasets["validation"]
                test_dict[file_path] = tokenized_datasets["test"]

        # concatenate all datasets and stream. Data from all languages willn stop streaming as soon as data from any language is exhausted.
        # If you want to keep streaming, change the stopping strategy to "last_exhausted"
        self.train_dataset = interleave_datasets(train_dict.values(), seed=SEED, stopping_strategy="first_exhausted")
        self.validation_dataset = interleave_datasets(validation_dict.values(), seed=SEED, stopping_strategy="first_exhausted")
        self.test_dataset = interleave_datasets(test_dict.values(), seed=SEED, stopping_strategy="first_exhausted")
        self.individual_validation_dataset = validation_dict


    def tokenize_group_function(self, examples):
        return self.vocab.tokenizer(examples["text"], return_special_tokens_mask=True)




class JointInputcorpus(object):
    def __init__(self,  language,
                        dataset_name,
                        tokenizer,
                        max_seq_length,
                        accelerator,
                        cache_dir,
                        model_type,
                        language_to_script_id: Optional[str] = None):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.language_to_script_id = language_to_script_id

        if dataset_name == "xnli":
            if language not in ["bn", "te"]:
                dataset = load_dataset(
                    "xnli",
                    language,
                    cache_dir=cache_dir)
            else:
                dataset = load_dataset(
                        "Divyanshu/indicxnli",
                        language,
                        cache_dir=cache_dir
                    )
            # Log a few random samples from the training set:
            train_data = dataset["train"]
            for index in random.sample(range(len(train_data)), 3):
                logger.info(f"Sample {index} of the training set: {train_data[index]}.")

            with accelerator.main_process_first():
                tokenized_datasets = dataset.map(
                self.preprocess,
                batched=True,
                desc=f"Running tokenizer on {dataset_name} and language {language} train dataset",
                fn_kwargs={"dataset_name": dataset_name})
                if self.language_to_script_id is not None and model_type == "routing":
                    tokenized_datasets = tokenized_datasets.map(dinsert_special_token_ids,  fn_kwargs={"script_id": self.language_to_script_id[language]} )

        elif dataset_name == "paws-x":
            dataset = load_dataset("paws-x", language, cache_dir=cache_dir)
            with accelerator.main_process_first():
                tokenized_datasets = dataset.map(
                self.preprocess,
                batched=True,
                desc=f"Running tokenizer on {dataset_name} train dataset",
                fn_kwargs={"dataset_name": dataset_name})
                if self.language_to_script_id is not None and model_type == "routing":
                    tokenized_datasets = tokenized_datasets.map(dinsert_special_token_ids,  fn_kwargs={"script_id": self.language_to_script_id[language]} )

        label_list = dataset["train"].unique("label")
        label_list.sort()  # Let's sort it for determinism
        self.num_labels = len(label_list)

        self.train_dataset = tokenized_datasets["train"]
        self.validation_dataset = tokenized_datasets["validation"]
        self.test_dataset = tokenized_datasets["test"]


    def preprocess(self, examples, dataset_name):
        if dataset_name == "xnli":
            return self.tokenizer(examples["premise"],
                                    examples["hypothesis"],
                                    padding="max_length",
                                    max_length=self.max_seq_length,
                                    truncation=True)

        elif dataset_name == "paws-x":
            return self.tokenizer(examples["sentence1"],
                                    examples["sentence2"],
                                        padding="max_length",
                                        max_length=self.max_seq_length,
                                        truncation=True)


