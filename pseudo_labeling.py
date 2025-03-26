import gc
import copy
import torch
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

from utils.dataset import NERDataset
from utils.dataloader import create_dataloader
from train import train_epoch, evaluate_epoch

def train_pseudo_labeling(model, optimizer, train_loader, val_loader, unlabeled_data, CONFIG):
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = "min", factor = 0.1, patience = 5)

    best_val_f1 = -float("inf")
    best_train_f1 = 0
    patience_counter = CONFIG.PATIENCE

    for epoch in range(CONFIG.EPOCHS):

        _, train_f1 = train_epoch(model, train_loader, optimizer, CONFIG)
        val_loss, val_f1 = evaluate_epoch(model, val_loader, CONFIG)

        scheduler.step(val_loss)

        # Save state of best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_train_f1 = train_f1
            patience_counter = CONFIG.PATIENCE
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter -= 1

        if patience_counter == 0:
            break  # Stop training if model doesn't improve

        # Generate pseudo-labels with trained model on unlabeled data
        pseudo_labels = generate_pseudo_labels(model, unlabeled_data, CONFIG)
        confidence_threshold = pseudo_labels["confidence_score"].quantile(CONFIG.CONFIDENCE_QUANTILE)

        def filter_tags(row):
            high_confidence = row["confidence_score"] > confidence_threshold
            # low_entropy = row["entropy"] < CONFIG.ENTROPY_THRESHOLD
            low_entropy = True
            representative = set(row["ner_tags"]) != {0}
            same_length = len(row["tokens"]) == len(row["ner_tags"])
            return high_confidence and low_entropy and representative and same_length

        labels_to_keep = pseudo_labels.apply(filter_tags, axis=1)
        good_pseudo_labels = pseudo_labels[labels_to_keep]
        pseudo_labels = pseudo_labels[~labels_to_keep]

        pseudo_dataset = NERDataset(good_pseudo_labels["tokens"].tolist(), good_pseudo_labels["ner_tags"].tolist())

        if epoch > CONFIG.PSEUDO_DELAY:
            existing_data = train_loader.dataset
            combined_dataset = ConcatDataset([existing_data, pseudo_dataset])
            train_loader = DataLoader(combined_dataset, CONFIG.BATCH_SIZE)
            print(f"Added {len(good_pseudo_labels)} rows of data")
        else:
            print("Early epoch")

    # Delete to clear up memory
    model.to("cpu")
    del optimizer, scheduler, model

    # Clear cache
    gc.collect()
    torch.cuda.empty_cache()

    return best_model_state, best_train_f1, best_val_f1


def generate_pseudo_labels(model, unlabeled_data, CONFIG):

    unlabeled_dataloader = create_dataloader(unlabeled_data, CONFIG, include_sentence=True)

    # Initialize lists to store pseudo-labels and confidence scores
    pseudo_sentences, pseudo_tags, pseudo_confidence_scores, entropy_scores = [], [], [], []

    for batch in unlabeled_dataloader:
        texts = batch["orginal_text"]
        texts = [text.split() for text in texts]
        del batch["orginal_text"]

        batch = {key : value.to(CONFIG.DEVICE) for key, value in batch.items()}

        with torch.no_grad():
            emissions, _ = model(**batch)
            predicted_tags = model.decode(emissions, batch["attention_mask"])

            # Compute sequence probabilities and entropy
            probs = F.softmax(emissions, dim=-1)
            sequence_confidence_scores, sequence_entropies = [], []
            for i, tags in enumerate(predicted_tags):
                token_confidence = [probs[i, j, tag].item() for j, tag in enumerate(tags)]
                token_entropy = -torch.sum(probs[i] * torch.log(probs[i] + 1e-9), dim=-1).cpu().numpy()

                seq_confidence = sum(token_confidence) / len(token_confidence)
                seq_entropy = sum(token_entropy) / len(token_entropy)
                sequence_confidence_scores.append(seq_confidence)
                sequence_entropies.append(seq_entropy)

            predicted_tags = [
                sequence[:CONFIG.MAX_SEQ_LEN] + [0] * max(0, CONFIG.MAX_SEQ_LEN - len(sequence))
                for sequence in predicted_tags
            ]

            # Trim predicted tags
            trimmed_predicted_tags = []
            word_counts = [len(text) for text in texts]
            for tag_seq, word_count in zip(predicted_tags, word_counts):
                tag_seq = tag_seq[1:-1]
                trimmed_predicted_tags.append(tag_seq[:word_count])

            pseudo_sentences.extend(texts)
            pseudo_tags.extend(trimmed_predicted_tags)
            pseudo_confidence_scores.extend(sequence_confidence_scores)
            entropy_scores.extend(sequence_entropies)

    pseudo_df = pd.DataFrame({
        "tokens": pseudo_sentences,
        "ner_tags": pseudo_tags,
        "confidence_score": pseudo_confidence_scores,
        "entropy": entropy_scores
    })

    return pseudo_df