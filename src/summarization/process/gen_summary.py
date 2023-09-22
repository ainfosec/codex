import re
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
from hdbscan import HDBSCAN
from sklearn import metrics
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise_distances
from torch import Tensor

warnings.filterwarnings("ignore")


def tokenizer(text):
    """
    Custom tokenizer that splits on whitespace to preserve single chars.
    :param text Text to tokenize
    :returns A list of tokens
    """
    return re.split("\\s+", text)


def enum_clusters(
    umap_vectors: np.ndarray, min_cluster_size: int, min_samples: int
) -> HDBSCAN:
    """
    Enumerate clusters from UMAP vectors.
    :param umap_vectors UMAP vectors to cluster
    :param min_cluster_size The minimum size of clusters
    :param min_samples The number of samples in a neighborhood for core points
    :returns An HDBSCAN object
    """
    hdb = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method="leaf",
    ).fit(umap_vectors)
    return hdb


def traverse_clusters(
    vectors: Tensor,
    umap_vectors: np.ndarray,
    tags: List[str],
    hdb: HDBSCAN,
    threshold: float,
    random_state: Optional[int] = None,
) -> Tuple[List[str], Optional[float], float]:
    """
    Traverse clusters to construct episode summary.
    :param vectors Vectors before dim. reduction
    :param umap_vectors UMAP vectors after dim. reduction
    :param tags Episode tags
    :param hdb HDBSCAN object
    :param threshold Cosine similarity value below which tags are added to summary tags
    :param random_state The seed used by the random number generator
    :returns Summary, Silhouette Coefficient and Global Cosine Similarity
    """
    clustered = hdb.labels_ >= 0
    cluster_ids = np.unique(hdb.labels_)
    num_ids = np.delete(cluster_ids, np.where(cluster_ids == -1))
    num_clusters = len(num_ids)
    summary_tags = []
    global_cos_sim = 0.0

    # Silhouette Coefficient
    sil_score = None
    if num_clusters > 1:
        sil_score = metrics.silhouette_score(
            umap_vectors[clustered], hdb.labels_[clustered]
        )

    for cluster in cluster_ids:
        if cluster == -1:
            continue

        num_data_points = np.count_nonzero(hdb.labels_ == cluster)
        timestamps = []
        cluster_tags = []
        cluster_umap_vecs = np.empty(
            shape=(num_data_points, 2), dtype="float32"
        )  # 2-dim.
        cluster_vecs = []  # 384-dim.
        vec_x = 0.0
        vec_y = 0.0
        vec_idx = 0

        for i in range(len(hdb.labels_)):
            if hdb.labels_[i] == cluster:
                cluster_umap_vecs[vec_idx][0] = np.asarray(
                    float(umap_vectors[i][0]), dtype="float32"
                )
                cluster_umap_vecs[vec_idx][1] = np.asarray(
                    float(umap_vectors[i][1]), dtype="float32"
                )
                vec_x += umap_vectors[i][0]
                vec_y += umap_vectors[i][1]
                cluster_vecs.append(vectors[i])
                timestamp, tag = tags[i].split(": ")
                timestamps.append(timestamp)
                cluster_tags.append(tag)
                vec_idx += 1

        # Calculate cluster cosine similarity with 384-dim. vectors
        cluster_cos_sim = 0.0
        vectors_centroid = sum(cluster_vecs)
        for i in range(len(cluster_vecs)):
            norm = np.linalg.norm(vectors_centroid.cpu())
            norm *= np.linalg.norm(cluster_vecs[i].cpu())
            dot = np.dot(vectors_centroid.cpu(), cluster_vecs[i].cpu())
            cluster_cos_sim += dot / norm
        cluster_cos_sim = cluster_cos_sim / len(cluster_vecs)
        global_cos_sim += cluster_cos_sim

        # 2-dim. cluster centroid for nearest distances calculation
        umap_centroid_x = vec_x / len(cluster_tags)
        umap_centroid_y = vec_y / len(cluster_tags)

        # Build summary elements
        topic_ngram = extract_topic_ngram(cluster_tags, random_state)
        centroid_idx = get_centroid_idx(
            cluster_umap_vecs, umap_centroid_x, umap_centroid_y
        )
        tag_matches = get_tag_matches(
            cluster_vecs,
            centroid_idx,
            topic_ngram,
            cluster_tags,
            timestamps,
            threshold,
        )
        summary_tags.extend(tag_matches)

    # Construct and print summary
    summary = construct_summary(summary_tags)

    # Compute global cosine similarity across all clusters
    global_cos_sim = global_cos_sim / num_clusters
    return summary, sil_score, global_cos_sim


def extract_topic_ngram(tags, random_state=None):
    """
    Extract Latent Dirichlet Allocation (LDA) topic ngram from a cluster.
    :param tags Tags in the cluster
    :param random_state The seed used by the random number generator
    :returns Topic ngram list
    """
    # Find shortest and longest ngrams in the cluster
    min_ngram = min(tags, key=lambda x: len(x.split()))
    max_ngram = max(tags, key=lambda x: len(x.split()))

    # Extract term frequencies for LDA model
    tf_vectorizer = CountVectorizer(
        lowercase=False,
        tokenizer=tokenizer,
        ngram_range=(len(min_ngram.split()), len(max_ngram.split())),
    )

    # Fit LDA model to the ngram vectors
    tf = tf_vectorizer.fit_transform(tags)
    lda = LatentDirichletAllocation(n_components=1, random_state=random_state)
    lda.fit(tf)
    tf_feature_names = tf_vectorizer.get_feature_names_out()

    # Extract cluster topic ngram
    for topic_idx, topic in enumerate(lda.components_):
        topic_ngram = [tf_feature_names[i] for i in topic.argsort()[:-2:-1]]
    return topic_ngram


def get_centroid_idx(umap_vectors, umap_centroid_x, umap_centroid_y):
    """
    Retrieve tag indices ordered closest to 2-dim. cluster centroid.
    :param umap_vectors UMAP vectors for cluster tags
    :param umap_centroid_x Cluster centroid's x-coordinate
    :param umap_centroid_y Cluster centroid's y-coordinate
    :returns An ndarray of tag indices by centroid proximity
    """
    centroid_vec = np.empty(shape=(1, 2), dtype="float32")
    centroid_vec[0][0] = np.asarray(umap_centroid_x, dtype="float32")
    centroid_vec[0][1] = np.asarray(umap_centroid_y, dtype="float32")
    nearest_distances = pairwise_distances(centroid_vec, umap_vectors, n_jobs=-1)
    top_idx = np.argsort(nearest_distances)
    centroid_idx = top_idx[0]
    return centroid_idx


def get_tag_matches(vectors, idx, ngram, tags, timestamps, threshold):
    """
    Find tag closest to cluster centroid that matches LDA topic ngram.
    Find any tags that are below the cos_sim threshold with the centroid tag.
    Fully extractive method.
    :param vectors 384-dim. embedding vectors for cosine similarities
    :param idx An ordered ndarray of tag indices closest to cluster centroid
    :param ngram LDA topic ngram
    :param tags List of cluster tags
    :param timestamps List of tag timestamps
    :param threshold Float of cosine similarity threshold
    :returns List of tag strings
    """
    tag_matches = []
    for i in range(len(idx)):
        if ngram[0] in tags[idx[i]]:
            tag_match = timestamps[idx[i]] + " " + tags[idx[i]]
            tag_idx = idx[i]
            tag_matches.append(tag_match)

            # Test for semantic diversity
            for j in range(len(idx)):
                norm = np.linalg.norm(vectors[tag_idx].cpu())
                norm *= np.linalg.norm(vectors[idx[j]].cpu())
                cos_sim = np.dot(vectors[tag_idx].cpu(), vectors[idx[j]].cpu()) / norm
                check = -1
                if cos_sim < threshold:
                    # Check if tag is already in tag_matches
                    substring = tags[idx[j]]
                    for k in range(len(tag_matches)):
                        check = tag_matches[k].find(substring)
                    if check == -1:
                        thresh_match = timestamps[idx[j]] + " *" + tags[idx[j]]
                        tag_matches.append(thresh_match)
                    break
            break
    return tag_matches


def construct_summary(tags) -> List[str]:
    """
    Construct and print summary by sorting summary tags by step number.
    :param tags List of summary tags
    :returns List of sorted summary tags
    """
    sorted_summary = sorted(tags, key=get_timestamp)
    for i in range(len(sorted_summary)):
        print(sorted_summary[i])
    return sorted_summary


def get_timestamp(tag: str) -> Union[int, Tuple[int, int]]:
    match = re.match("^(\d+) -- (\d+) (.+)", tag)
    if match:
        start = int(match.group(1))
        end = int(match.group(2))
        return (start, end)
    return int(tag.split()[0])
