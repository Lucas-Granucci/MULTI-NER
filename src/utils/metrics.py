import torch
from torcheval.metrics.functional import multiclass_f1_score

def f1_score(masked_predictions, masked_labels, num_labels):

    f1_score_tensor = multiclass_f1_score(masked_predictions, masked_labels, num_classes=num_labels)

    return f1_score_tensor.cpu().numpy().item()