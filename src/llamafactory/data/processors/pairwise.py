# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from .processor_utils import infer_seqlen


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ...hparams import DataArguments
    from ..mm_plugin import ImageInput, VideoInput
    from ..template import Template


logger = get_logger(__name__)


def _encode_pairwise_example(
    prompt: Sequence[Dict[str, str]],
    response: Sequence[Dict[str, str]],
    system: Optional[str],
    tools: Optional[str],
    images: Sequence["ImageInput"],
    videos: Sequence["VideoInput"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    cutoff_len: int,
) -> Tuple[List[int], List[int], List[int], List[int]]:
    chosen_messages = template.mm_plugin.process_messages(prompt + [response[0]], images, videos, processor)
    rejected_messages = template.mm_plugin.process_messages(prompt + [response[1]], images, videos, processor)
    prompt_ids, chosen_ids = template.encode_oneturn(tokenizer, chosen_messages, system, tools)
    _, rejected_ids = template.encode_oneturn(tokenizer, rejected_messages, system, tools)

    if template.efficient_eos:
        chosen_ids += [tokenizer.eos_token_id]
        rejected_ids += [tokenizer.eos_token_id]

    prompt_ids, _ = template.mm_plugin.process_token_ids(prompt_ids, None, images, videos, tokenizer, processor)
    # consider the response is more important
    source_len, target_len = infer_seqlen(len(prompt_ids), max(len(chosen_ids), len(rejected_ids)), cutoff_len)
    prompt_ids = prompt_ids[:source_len]
    chosen_ids = chosen_ids[:target_len]
    rejected_ids = rejected_ids[:target_len]

    chosen_input_ids = prompt_ids + chosen_ids
    chosen_labels = [IGNORE_INDEX] * source_len + chosen_ids
    rejected_input_ids = prompt_ids + rejected_ids
    rejected_labels = [IGNORE_INDEX] * source_len + rejected_ids
    return chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels


def preprocess_pairwise_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[Any]]:
    # build input pairs with format `<bos> X`, `Y1 <eos>` and `Y2 <eos>`
    model_inputs = defaultdict(list)
    for lang in ["source", "target"]:
        for i in range(len(examples[f"_{lang}_prompt"])):
            if len(examples[f"_{lang}_prompt"][i]) % 2 != 1 or len(examples[f"_{lang}_response"][i]) < 2:
                logger.warning("Dropped invalid example: {}".format(examples[f"_{lang}_prompt"][i] + examples[f"_{lang}_response"][i]))
                continue

            chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels = _encode_pairwise_example(
                prompt=examples[f"_{lang}_prompt"][i],
                response=examples[f"_{lang}_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                template=template,
                tokenizer=tokenizer,
                processor=processor,
                cutoff_len=data_args.cutoff_len,
            )
            model_inputs[f"{lang}_chosen_input_ids"].append(chosen_input_ids)
            model_inputs[f"{lang}_chosen_attention_mask"].append([1] * len(chosen_input_ids))
            model_inputs[f"{lang}_chosen_labels"].append(chosen_labels)
            model_inputs[f"{lang}_rejected_input_ids"].append(rejected_input_ids)
            model_inputs[f"{lang}_rejected_attention_mask"].append([1] * len(rejected_input_ids))
            model_inputs[f"{lang}_rejected_labels"].append(rejected_labels)
            if lang == 'source':
                model_inputs["images"].append(examples["_images"][i])
                model_inputs["videos"].append(examples["_videos"][i])
    
    return model_inputs


def print_pairwise_dataset_example(example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer") -> None:
    source_valid_chosen_labels = list(filter(lambda x: x != IGNORE_INDEX, example["source_chosen_labels"]))
    source_valid_rejected_labels = list(filter(lambda x: x != IGNORE_INDEX, example["source_rejected_labels"]))
    print("source_chosen_input_ids:\n{}".format(example["source_chosen_input_ids"]))
    print("source_chosen_inputs:\n{}".format(tokenizer.decode(example["source_chosen_input_ids"], skip_special_tokens=False)))
    print("source_chosen_label_ids:\n{}".format(example["source_chosen_labels"]))
    print("source_chosen_labels:\n{}".format(tokenizer.decode(source_valid_chosen_labels, skip_special_tokens=False)))
    print("source_rejected_input_ids:\n{}".format(example["source_rejected_input_ids"]))
    print("source_rejected_inputs:\n{}".format(tokenizer.decode(example["source_rejected_input_ids"], skip_special_tokens=False)))
    print("source_rejected_label_ids:\n{}".format(example["source_rejected_labels"]))
    print("source_rejected_labels:\n{}".format(tokenizer.decode(source_valid_rejected_labels, skip_special_tokens=False)))

    target_valid_chosen_labels = list(filter(lambda x: x != IGNORE_INDEX, example["target_chosen_labels"]))
    target_valid_rejected_labels = list(filter(lambda x: x != IGNORE_INDEX, example["target_rejected_labels"]))
    print("target_chosen_input_ids:\n{}".format(example["target_chosen_input_ids"]))
    print("target_chosen_inputs:\n{}".format(tokenizer.decode(example["target_chosen_input_ids"], skip_special_tokens=False)))
    print("target_chosen_label_ids:\n{}".format(example["target_chosen_labels"]))
    print("target_chosen_labels:\n{}".format(tokenizer.decode(target_valid_chosen_labels, skip_special_tokens=False)))
    print("target_rejected_input_ids:\n{}".format(example["target_rejected_input_ids"]))
    print("target_rejected_inputs:\n{}".format(tokenizer.decode(example["target_rejected_input_ids"], skip_special_tokens=False)))
    print("target_rejected_label_ids:\n{}".format(example["target_rejected_labels"]))
    print("target_rejected_labels:\n{}".format(tokenizer.decode(target_valid_rejected_labels, skip_special_tokens=False)))
