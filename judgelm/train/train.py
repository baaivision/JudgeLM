# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
from dataclasses import dataclass, field
import json
import math
import pathlib
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
import transformers

from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother

import sys
from pathlib import Path # if you haven't already done so
file = Path(__file__).resolve()
root = file.parents[2]
sys.path.append(str(root))
print(sys.path)

from judgelm.conversation import SeparatorStyle
from judgelm.model.model_adapter import get_conversation_template
from judgelm.utils import jlload, jload
from judgelm.llm_judge.common import conv_judge_pair, conv_judge_pair_w_reference


IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    swap_aug_ratio: float = -1.0
    ref_drop_ratio: float = -1.0


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = get_conversation_template("vicuna")
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO

    # Mask targets. Only compute loss on the assistant outputs.
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        turns = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, turn in enumerate(turns):
            if turn == "":
                break
            turn_len = len(tokenizer(turn).input_ids)

            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            # "-2" is hardcoded for the LLaMA tokenizer to make the offset correct.
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            # Ignore the user instructions
            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += turn_len

        target[cur_len:] = IGNORE_TOKEN_ID

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                rank0_print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class AlpacaSupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        PROMPT_DICT = {
            "prompt_input": (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
            ),
            "prompt_no_input": (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Response:"
            ),
        }

        rank0_print("Loading data...")
        list_data_dict = jload(data_path)

        rank0_print("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        rank0_print("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def swap_first_two_integers(s):
    # find the first space
    first_space_index = s.find(' ')
    if first_space_index != -1:
        # find the second space
        second_space_index = s.find('\n', first_space_index + 1)
        if second_space_index != -1:
            # swap the first two integers
            new_s = s[first_space_index + 1:second_space_index] + ' ' + s[:first_space_index] + '\n' + s[
                                                                                                    second_space_index + 1:]
            return new_s

    return s

class LazyJudgeSupervisedDataset(AlpacaSupervisedDataset):

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, swap_aug_ratio: float, ref_drop_ratio: float):
        Dataset.__init__(self)

        self.tokenizer = tokenizer
        self.swap_aug_ratio = swap_aug_ratio
        self.ref_drop_ratio = ref_drop_ratio

        rank0_print("Loading data...")
        self.list_data_dict = jlload(data_path)

        rank0_print("Formatting inputs...")

        list_data_dict_cleaned = []
        for data in self.list_data_dict:
            if data['text'] == 'error':
                continue
            list_data_dict_cleaned.append(data)
        self.list_data_dict = list_data_dict_cleaned

    def __len__(self):
        return len(self.list_data_dict)

    def swap_first_two_integers(self, s):
        # find the first space
        first_space_index = s.find(' ')
        if first_space_index != -1:
            # find the second space
            second_space_index = s.find('\n', first_space_index + 1)
            if second_space_index != -1:
                # swap the first two integers
                new_s = s[first_space_index + 1:second_space_index] + ' ' + s[:first_space_index] + '\n' + s[
                                                                                                           second_space_index + 1:]
                return new_s
        return s

    def add_reference(self, data):
        data['text'] = data['text_w_reference']
        data['score'] = data['score_w_reference']
        return data

    def swap_aug(self, data):
        data['answer1_body'], data['answer2_body'] = data['answer2_body'], data['answer1_body']
        data['answer1_id'], data['answer2_id'] = data['answer2_id'], data['answer1_id']
        data['score'] = data['score'][::-1]
        data['answer1_model_id'], data['answer2_model_id'] = data['answer2_model_id'], data['answer1_model_id']
        data['answer1_metadata'], data['answer2_metadata'] = data['answer2_metadata'], data['answer1_metadata']

        data['text'] = self.swap_first_two_integers(data['text'])
        data['text'] = data['text'].replace('Assistant 1', 'Assistant X')
        data['text'] = data['text'].replace('Assistant 2', 'Assistant 1')
        data['text'] = data['text'].replace('Assistant X', 'Assistant 2')

        return data

    def get_data_sample(self, data, conv, ref_drop_flag):
        template = conv.prompt_template

        if ref_drop_flag:
            data_sample = conv.system + '\n' + template.format(question=data['question_body'],
                                                               answer_1=data['answer1_body'],
                                                               answer_2=data['answer2_body'],
                                                               prompt=conv.prompt) + conv.appendix
        else:
            data_sample = conv.system + '\n' + template.format(question=data['question_body'],
                                                               reference=data['reference']['text'],
                                                               answer_1=data['answer1_body'],
                                                               answer_2=data['answer2_body'],
                                                               prompt=conv.prompt) + conv.appendix
        return data_sample

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        data = self.list_data_dict[i]
        conv = conv_judge_pair.copy()

        # if ref_drop_ratio >= -0.5, then replace 'text' with 'text_w_reference'
        ref_drop_flag = True
        if self.ref_drop_ratio >= -0.5 and np.random.rand() < self.ref_drop_ratio:
            ref_drop_flag = False
            data = self.add_reference(data)
            conv = conv_judge_pair_w_reference.copy()


        if self.swap_aug_ratio >= -0.5 and np.random.rand() < self.swap_aug_ratio:
            data = self.swap_aug(data)

        data_sample = self.get_data_sample(data, conv, ref_drop_flag)

        target = f"{data['text']}{self.tokenizer.eos_token}"
        example = data_sample + target
        example_tokenized, source_tokenized = [_tokenize_fn(strings, self.tokenizer) for strings in ([example], [data_sample])]
        input_ids = example_tokenized["input_ids"]
        label = copy.deepcopy(input_ids)
        source_len = source_tokenized["input_ids_lens"][0]
        label[0][:source_len] = IGNORE_TOKEN_ID

        return dict(input_ids=input_ids[0], labels=label[0])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_TOKEN_ID)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazyJudgeSupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, swap_aug_ratio=data_args.swap_aug_ratio, ref_drop_ratio=data_args.ref_drop_ratio)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

def translate_params_from_str_to_bool(params):
    params_class = type(params)
    params = vars(params)
    for key in params:
        # check if the value is a string
        if not isinstance(params[key], str):
            continue
        if params[key].lower() == "true":
            params[key] = True
        elif params[key].lower() == "false":
            params[key] = False
        elif params[key].lower() == "none":
            params[key] = None

    return params_class(**params)

def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # set data_args from str to bool
    data_args = translate_params_from_str_to_bool(data_args)


    local_rank = training_args.local_rank
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    model.config.use_cache = False

    # Set RoPE scaling factor
    orig_ctx_len = getattr(model.config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = math.ceil(training_args.model_max_length / orig_ctx_len)
        model.config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    # [done]: fix 'RuntimeError: Expected q_dtype == torch::kFloat16 || (is_sm8x && q_dtype == torch::kBFloat16) to be true, but got false.'
    model = model.to(torch.bfloat16) if training_args.bf16 else model.to(torch.float16)

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    model.config.use_cache = True
    trainer.save_state()
    # wait 一段时间
    import time
    time.sleep(100)

    # todo: debug
    if training_args.deepspeed is not None:
        trainer.save_model(output_dir=training_args.output_dir)
        # wait 一段时间
        import time
        time.sleep(100)
        # fp32 saving
        # trainer.deepspeed.save_checkpoint(training_args.output_dir)
        # fp16 saving
        WEIGHTS_NAME = "pytorch_model.bin"
        trainer.deepspeed.save_16bit_model(training_args.output_dir, WEIGHTS_NAME)
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    # wait 一段时间
    import time
    time.sleep(100)

    if trainer.is_world_process_zero():
        print("Saving tokenizer...")
        print(trainer)
        print("Saving tokenizer...")
        print(training_args)



if __name__ == "__main__":
    train()
