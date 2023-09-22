import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModel, AutoTokenizer

from summarization.process.dim_reduce import umap_embed
from summarization.process.encode import encode_tags, get_tags
from summarization.process.gen_summary import enum_clusters, traverse_clusters
from summarization.process.rouge import rouge_scoring
from summarization.utils import MODEL_DIR, count_parameters, save_files
from utils import Tee


def main(args: argparse.Namespace) -> None:
    tee: Path = args.output.expanduser()
    with contextlib.redirect_stdout(Tee(sys.stdout, open(tee, "w"))):
        semantic_sum(args)


def semantic_sum(args: argparse.Namespace) -> None:
    # Extract CLI arguments.
    dataset: Path = args.dataset.expanduser()
    max_length: int = args.max_length
    save_vecs: bool = args.save_vecs
    n_components: int = args.n_components
    n_neighbors: int = args.n_neighbors
    min_cluster_size: int = args.min_cluster_size
    min_samples: int = args.min_samples
    sum_threshold: float = args.sum_threshold
    seed: Optional[int] = args.seed

    # GPU or CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModel.from_pretrained(MODEL_DIR)
    model.to(device)
    _, params = count_parameters(model)
    print("Initializing semantic cluster-based summarizer...")
    print("Language Model parameters : {}".format(params))
    print(f"Using {device}")
    print("\n", end="")

    # Load json factual and counterfactual tags collection
    with open(dataset) as f:
        data = json.load(f)

    # Summarize episode factual and counterfactuals with semantic clustering
    for episode in range(len(data)):
        # Factual processing
        print("####  EPISODE {}  ####\n".format(str(episode)))
        print("FACTUAL")
        factual_tags = get_tags(data, episode, "factual", None)
        factual_vecs = encode_tags(model, tokenizer, device, factual_tags, max_length)
        u_emb = umap_embed(factual_vecs, n_components, n_neighbors, random_state=seed)
        hdb = enum_clusters(u_emb, min_cluster_size, min_samples)
        factual_summary, sil_score, global_cos_sim = traverse_clusters(
            factual_vecs, u_emb, factual_tags, hdb, sum_threshold, random_state=seed
        )
        print("\n", end="")

        # Clustering eval metrics
        if sil_score is not None:
            print("Silhouette Coefficient : {:.3f}".format(sil_score))
        else:
            print("Silhouette Coefficient : N/A")
        print("Global Cosine Similarity : {:.3f}".format(global_cos_sim))
        print("\n", end="")

        if save_vecs:
            save_files(str(episode), factual_tags, factual_vecs, "factual")

        # Counterfactuals processing
        cfx_start = data[episode].get("cfx_start", [])
        num_cfx = len(cfx_start)
        for cfx in range(num_cfx):
            print("CFX {}".format(str(cfx)))
            cfx_tags = get_tags(data, episode, "cfx", cfx)
            cfx_vecs = encode_tags(model, tokenizer, device, cfx_tags, max_length)
            u_emb = umap_embed(cfx_vecs, n_components, n_neighbors, random_state=seed)
            hdb = enum_clusters(u_emb, min_cluster_size, min_samples)
            cfx_summary, sil_score, global_cos_sim = traverse_clusters(
                cfx_vecs, u_emb, cfx_tags, hdb, sum_threshold, random_state=seed
            )
            rouge_1_f, rouge_2_f = rouge_scoring(factual_summary, cfx_summary)
            print("\n", end="")

            # Clustering and summary eval metrics
            print("Silhouette Coefficient : {:.3f}".format(sil_score))
            print("Global Cosine Similarity : {:.3f}".format(global_cos_sim))
            print("ROUGE-1-F : {:.2f}".format(rouge_1_f))
            print("ROUGE-2-F : {:.2f}".format(rouge_2_f))
            print("\n", end="")

            if save_vecs:
                save_files(str(episode) + "-" + str(cfx), cfx_tags, cfx_vecs, "cfx")


if __name__ == "__main__":
    main()
