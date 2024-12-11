import torch
from torcheval.metrics.functional import multiclass_f1_score


def f1_score(masked_predictions, masked_labels, num_labels):
    f1_score_tensor = multiclass_f1_score(
        masked_predictions, masked_labels, num_classes=num_labels
    )

    return f1_score_tensor.cpu().numpy().item()


def prepare_labels(decoded_emissions, labels, attention_mask, label_pad_idx=-100):
    # Flatten for direct comparison
    predicted_labels = torch.cat([torch.tensor(emission, device=labels.device) for emission in decoded_emissions])
    labels = labels.flatten()
    attention_mask = attention_mask.flatten()

    mask = (attention_mask == 1) & (labels != label_pad_idx)

    masked_predictions = predicted_labels[mask]
    masked_labels = labels[mask]

    return masked_predictions, masked_labels
