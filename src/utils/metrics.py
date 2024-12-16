import torch
from torcheval.metrics.functional import multiclass_f1_score
from sklearn.metrics import classification_report


def f1_score(masked_predictions, masked_labels, num_labels):
    f1_score_tensor = multiclass_f1_score(
        masked_predictions, masked_labels, num_classes=num_labels
    )

    return f1_score_tensor.cpu().numpy().item()


def print_classification_report(masked_predictions, masked_labels):
    # Convert tensors to numpy arrays for sklearn
    masked_predictions_np = [mp.cpu().numpy() for mp in masked_predictions]
    masked_labels_np = [ml.cpu().numpy() for ml in masked_labels]

    # Generate classification report
    report = classification_report(
        masked_labels_np, masked_predictions_np, output_dict=True
    )

    for key, value in report.items():
        print(f"{key.capitalize()}:")
        if isinstance(value, dict):
            for metric, metric_value in value.items():
                print(f"  {metric.capitalize():<12}: {metric_value:.4f}")
        else:
            print(f"  {value:.4f}")
        print()


def prepare_labels(decoded_emissions, labels, attention_mask, label_pad_idx=-100):
    # Flatten for direct comparison
    predicted_labels = torch.cat(
        [torch.tensor(emission, device=labels.device) for emission in decoded_emissions]
    )
    labels = labels.flatten()
    attention_mask = attention_mask.flatten()

    mask = (attention_mask == 1) & (labels != label_pad_idx)

    masked_predictions = predicted_labels[mask]
    masked_labels = labels[mask]

    return masked_predictions, masked_labels
