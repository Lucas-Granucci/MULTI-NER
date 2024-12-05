import torch
from torcheval.metrics.functional import multiclass_f1_score


def f1_score(masked_predictions, masked_labels, num_labels):
    f1_score_tensor = multiclass_f1_score(
        masked_predictions, masked_labels, num_classes=num_labels
    )

    return f1_score_tensor.cpu().numpy().item()


def prepare_labels(emissions, labels, attention_mask, label_pad_idx=-100):
    # Prepare outputs for F1-scoring
    predicted_labels = emissions.argmax(dim=-1)

    # Flatten for direct comparison
    predicted_labels = torch.flatten(predicted_labels)
    labels = torch.flatten(labels)
    attention_mask = torch.flatten(attention_mask)

    mask = (attention_mask == 1) & (labels != label_pad_idx)

    masked_predictions = predicted_labels[mask]
    masked_labels = labels[mask]

    return masked_predictions, masked_labels
