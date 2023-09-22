from utils import OUTPUT_DIR, ROOT_DIR

RESOURCES = ROOT_DIR / "resources"
MODEL_DIR = RESOURCES / "model"
VECS_DIR = OUTPUT_DIR / "episode_vecs"


def count_parameters(model):
    """
    Show number of trainable and total model parameters.
    :param model A language model
    :returns Two ints
    """
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return train_params, total_params


def save_files(episode, tags, tag_vecs, ep_type):
    """
    Save tags and tag vectors to files for debugging.
    :param episode Episode number
    :param tags A list of episode tags
    :param tag_vecs A list of tag vectors
    :param ep_type Factual or Counterfactual episode
    """
    output_vecs = {}
    for i in range(len(tags)):
        output_vecs[i, tags[i]] = tag_vecs[i]

    vecs_path = f"{VECS_DIR}/{ep_type}_ep_{episode}_ids_vecs.txt"
    tags_path = f"{VECS_DIR}/{ep_type}_ep_{episode}_ids_tags.txt"
    with open(vecs_path, "w") as ids_vecs, open(tags_path, "w") as ids_tags:
        for key, value in output_vecs.items():
            print("tagID_" + str(key[0]), end="", file=ids_vecs)
            print("tagID_" + str(key[0]) + " " + key[1], end="", file=ids_tags)
            for i in range(len(value)):
                print(" {:.4f}".format(value[i]), end="", file=ids_vecs)
            print("\n", end="", file=ids_vecs)
            print("\n", end="", file=ids_tags)
