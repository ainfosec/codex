import re

from rouge import Rouge

rouge = Rouge()


def rouge_scoring(factual_sum, cfx_sum):
    """
    Calculate ROUGE scores for factual (GOLD REF) and counterfactual (CANDIDATE) summaries.
    :param factual_sum List of sorted factual summary tags
    :param cfx_sum List of sorted counterfactual summary tags
    :returns ROUGE-1-F and ROUGE-2-F scores
    """
    for i in range(len(factual_sum)):
        factual_sum[i] = re.sub(r"^\d+ \**", "", factual_sum[i])
    factual_sum = (" ").join(factual_sum)

    for i in range(len(cfx_sum)):
        cfx_sum[i] = re.sub(r"^\d+ \**", "", cfx_sum[i])
    cfx_sum = (" ").join(cfx_sum)

    rouge_scores = rouge.get_scores(cfx_sum, factual_sum)
    rouge_1_f = 100 * float(rouge_scores[0]["rouge-1"]["f"])
    rouge_2_f = 100 * float(rouge_scores[0]["rouge-2"]["f"])
    return rouge_1_f, rouge_2_f
