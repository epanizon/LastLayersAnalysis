from datasets import load_dataset, load_from_disk
import torch
from functools import partial
from datasets.utils.logging import disable_progress_bar
import numpy as np

import torchvision.datasets as tv_datasets
from torchvision import transforms


disable_progress_bar()


def encode_func(example, tokenizer, max_seq_length, text_field, choices_field, label_field):
    # multiple answering convention
    ilett = ['A. ', 'B. ', 'C. ', 'D. ', 'E. ','F. ', 'G. ', 'H. ', 'I. ', 'J. ', 'K. ', 'L. ']
    example_text = example[text_field]
    example_choices = example[choices_field]
    for ic, c in enumerate(example_choices):
        example_text += f'\n'+ilett[ic]+c
    example_text += '\nAnswer: '
    
    tokenized_example = tokenizer(
        example_text.strip(),
        add_special_tokens=False,
        return_tensors="pt",
        max_length=max_seq_length,
        truncation=True,
    )
    input_ids = tokenized_example.input_ids
    correct_id = example[label_field]
    #labels = tokenizer(ilett[correct_id][0], add_special_tokens=False).input_ids
    labels = input_ids.clone()
    attention_mask = tokenized_example.attention_mask
    # in the output they will be converted to lists but at least we can apply flatten
    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }

def get_text_dataset(
    filepath=None,
    tokenizer=None,
    max_seq_length=2048,
    num_processes=1,
    text_field='question',
    choices_field='choices',
    label_field='answer',
):
    raw_dataset = load_from_disk(
        filepath)

    encode_function = partial(
        encode_func,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        text_field=text_field,
        choices_field=choices_field,
        label_field=label_field
    )

    tokenized_dataset = raw_dataset.map(
        encode_function,
        batched=False,
        num_proc=num_processes,
        load_from_cache_file=False,
        remove_columns=[
            name
            for name in raw_dataset.column_names
            if name not in ["input_ids", "labels", "attention_mask"]
        ],
    )
    # the output is always list of lists
    tokenized_dataset.set_format(type="pt")

    return tokenized_dataset
