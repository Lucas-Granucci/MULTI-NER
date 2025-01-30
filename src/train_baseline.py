import json
from training.cv_train import cv_train
from training.train_configs import model_configs
from preprocessing.dataloader import load_data
from preprocessing.dataset import NERDataset
from models.BertCrf import BertCrf

# Define languages
languages = {
    "fo": load_data("data/labeled/fo_data.csv"),
    "co": load_data("data/labeled/co_data.csv"),
    "hsb": load_data("data/labeled/hsb_data.csv"),
    "bh": load_data("data/labeled/bh_data.csv"),
    "cv": load_data("data/labeled/cv_data.csv"),
    "mg": load_data("data/labeled/mg_data.csv"),
}

# Define model
model_name = "BERT-CRF"
baseline_model = BertCrf

baseline_model_performance = {}

# Iterate over languages
for language, lang_df in languages.items():
    print(f"Testing: {model_name} on language: '{language}'")

    # Train and evaluate model
    val_f1, train_f1 = cv_train(
        Model=baseline_model,
        dataframe=lang_df,
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
with open("src/experiments/results/baseline_model_performance.json", "w") as outfile:
    json.dump(baseline_model_performance, outfile)
