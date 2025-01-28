import json
from training.cv_train import cv_train
from training.train_configs import model_configs
from preprocessing.dataloader import load_data
from preprocessing.dataset import NERDataset
from models.BertCrf import BertCrf

# Define languages
low_resource_languages = {
    "is": load_data("data/raw/is_data.csv").head(4000),
    "en": load_data("data/raw/en_data.csv").head(4000),
}

high_resource_languages = {"en": load_data("data/raw/en_data.csv")}

# Define model
model_name = "BERT-CRF"
baseline_model = BertCrf

baseline_model_performance = {}

# Iterate over languages
for language, lang_df in low_resource_languages.items():
    print(f"Testing: {model_name} on language: {language}")

    # Create NER dataset for language
    dataset = NERDataset(
        texts=lang_df["tokens"].to_list(), tags=lang_df["ner_tags"].to_list()
    )

    # Train and evaluate model
    val_f1, train_f1 = cv_train(
        Model=baseline_model,
        dataset=dataset,
        k_splits=5,
        config=model_configs[model_name],
    )

    # Save model results
    baseline_model_performance[language] = {
        "train_f1": train_f1,
        "val_f1": val_f1,
    }

    print(f"{model_name} on {language} -- Val F1: {val_f1}, Train F1: {train_f1}")

# Save results to json
with open("src/experiments/results/test_model_performance.json", "w") as outfile:
    json.dump(baseline_model_performance, outfile)
