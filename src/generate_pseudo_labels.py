import pandas as pd
import torch
from tqdm import tqdm
from preprocessing.dataloader import LanguageDataManager
from preprocessing.dataset import U_NERDataset
from models.BertBilstmCrf import BertBilstmCrf
from torch.utils.data import DataLoader
from config import ExperimentConfig
import torch.nn.functional as F

# Configuration for pseudo-labeling
pseudo_labeling_config = ExperimentConfig(
    num_tags=7,
    batch_size=48,
    patience=5,
    epochs=20,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    bert_learning_rate=0.00003,
    lstm_learning_rate=0.005,
    crf_learning_rate=0.00005,
    seed=42,
    low_resource_base_count=240,
    results_dir="src/experiments/results/objective_IV",
    model_dir="src/models/pretrained",
    logging_dir="src/experiments/logging/objective_IV",
)

def generate_labels(teacher_model, unlabeled_df, config):
    # Create unlabeled NER dataset
    dataset = U_NERDataset(texts=unlabeled_df["sentences"].to_list())
    unlabeled_dataloader = DataLoader(dataset, batch_size=config.batch_size)

    teacher_model.to(config.device)
    teacher_model.eval()

    # Initialize lists to store pseudo-labels and confidence scores
    pseudo_sentences = []
    pseudo_tags = []
    pseudo_confidence_scores = []

    for batch in tqdm(unlabeled_dataloader, desc="Generating pseudo-labels"):
        sentences = batch["sentence"]
        word_counts = batch["sentence_length"]

        del batch["sentence_length"]
        del batch["sentence"]

        # Move data to device
        batch = {key: value.to(config.device) for key, value in batch.items()}

        emissions, _ = teacher_model(**batch)

        # Decode and get predicted tags
        predicted_tags = teacher_model.decode(emissions, batch["mask"])

        # Compute sequence probabilities
        probs = F.softmax(emissions, dim=-1)
        sequence_confidence_scores = []
        for i, tags in enumerate(predicted_tags):
            token_confidence = [probs[i, j, tag].item() for j, tag in enumerate(tags)]
            seq_confidence = sum(token_confidence) / len(token_confidence)
            sequence_confidence_scores.append(seq_confidence)

        predicted_tags = [
            sequence[:dataset.MAX_LEN] + [0] * max(0, dataset.MAX_LEN - len(sequence)) 
            for sequence in predicted_tags
        ]

        # Trim predicted tags
        trimmed_predicted_tags = []
        for tag_seq, word_count in zip(predicted_tags, word_counts):
            tag_seq = tag_seq[1:-1]
            trimmed_predicted_tags.append(tag_seq[:word_count])

        pseudo_sentences.extend([sent.split(" ") for sent in sentences])
        pseudo_tags.extend(trimmed_predicted_tags)
        pseudo_confidence_scores.extend(sequence_confidence_scores)

    pseudo_df = pd.DataFrame({
        "tokens": pseudo_sentences,
        "ner_tags": pseudo_tags,
        "confidence_score": pseudo_confidence_scores
    })

    return pseudo_df

# Load data
data_manager = LanguageDataManager()
low_resource_languages = ["mg", "fo", "co", "hsb", "bh", "cv"]

language_data = data_manager.load_languages(low_resource_languages)
unlabeled_data = {
    lang: data_manager.load_unlabeled_data(f"data/unlabeled/{lang}_texts.txt")
    for lang in low_resource_languages
}

# Generate pseudo-labels for each language
for lang in low_resource_languages:
    teacher_model = BertBilstmCrf(num_tags=pseudo_labeling_config.num_tags)
    teacher_model.load_state_dict(
        torch.load(
            f"{pseudo_labeling_config.model_dir}/{lang}_baseline_pretrained.pth",
            weights_only=True,
        )
    )

    pseudo_df = generate_labels(teacher_model, unlabeled_data[lang], pseudo_labeling_config)
    pseudo_df.to_csv(f"data/pseudo_labels/{lang}_pseudo_labels.csv", index=False)
