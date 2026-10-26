"""Set Recall needed for validation; empty Ground Truth remains undefined."""


def recall(result_ids, ground_truth_ids):
    truth = set(ground_truth_ids)
    return len(set(result_ids) & truth) / len(truth) if truth else None
