import re
from typing import List

import torch


def flatten(nested: List) -> List:
    y = []
    for x in nested:
        if isinstance(x, list):
            y.extend(flatten(x))
        else:
            y.append(x)
    return y


def get_tags(data, episode, ep_type, cfx) -> List[str]:
    """
    Retrieve episode tags.
    :param data Source data
    :param episode Episode number
    :param ep_type Factual or Counterfactual episode type
    :param cfx Counterfactual number for an episode
    :returns List of episode tags
    """
    if ep_type == "factual":
        return flatten(data[episode]["ep_tags"])

    if ep_type == "cfx":
        return flatten(data[episode]["cfx_tags"][cfx])


def encode_tags(model, tokenizer, device, tags, max_length):
    """
    Encode episode tags as vectors.
    :param model A language model
    :param tokenizer A tokenizer
    :param device GPU or CPU device
    :param tags A list of tags
    :param max_length Tokenizer max sequence length
    :returns List of tag vectors
    """
    tags_no_step = []
    for i in range(len(tags)):
        # This regex strips the tag of the timestamp only if the
        # timestamp is a single integer followed by a colon. For
        # environments like StarCraft II, where timestamps have both
        # a start and end timestep, the tag is left unchanged.
        tags_no_step.append(re.sub(r"^\d+\: ", "", tags[i]))
    input_tokens = tokenizer(
        tags_no_step,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_tokens.to(device)

    with torch.no_grad():
        model_output = model(**input_tokens)
    tag_vecs = mean_pooling(model_output, input_tokens["attention_mask"])
    return tag_vecs


def mean_pooling(model_output, attention_mask):
    """
    Average token embeddings.
    Mean Pooling takes attention mask into account for correct averaging.
    First element of model_output contains all token embeddings.
    :param model_output Input token embeddings
    :param attention_mask Attention mask for input tokens
    :returns Averaged token embeddings
    """
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask
