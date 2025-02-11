import json
from training.train_evaluate import train_evaluate
from training.train_configs import model_configs
from preprocessing.dataloader import load_data
from models.BertBilstmCrf import BertBilstmCrf
from utils import set_seed

set_seed(42)

# Define languages
languages = {
    "mg": load_data("data/labeled/mg_data.csv"),
    "fo": load_data("data/labeled/fo_data.csv"),
    "co": load_data("data/labeled/co_data.csv"),
    "hsb": load_data("data/labeled/hsb_data.csv"),
    "bh": load_data("data/labeled/bh_data.csv"),
    "cv": load_data("data/labeled/cv_data.csv"),
}

# Define model
model_name = "BERT-Bilstm-CRF"
baseline_model = BertBilstmCrf

baseline_model_performance = {}
baseline_model_logging = {}

# Iterate over languages
for language, lang_df in languages.items():
    print(f"Testing: {model_name} on language: '{language}'")

    # Train and evaluate model
    train_f1, test_f1, logging_results = train_evaluate(
        ModelClass=baseline_model,
        dataframe=lang_df,
        config=model_configs[model_name],
        save_model=f"src/models/pretrained/{language}_baseline_pretrained.pth",
        verbose=True,
    )

    # Save model results
    baseline_model_performance[language] = {
        "train_f1": train_f1,
        "test_f1": test_f1,
    }

    # Save logging results
    baseline_model_logging[language] = logging_results

    print(f"{model_name} on {language} -- Test F1: {test_f1}, Train F1: {train_f1}")

# Save results to json
with open(
    "src/experiments/results/objective_III/baseline_model_performance.json", "w"
) as outfile:
    json.dump(baseline_model_performance, outfile)

# Save logging to json
with open(
    "src/experiments/results/objective_III/baseline_model_logging.json", "w"
) as outfile:
    json.dump(baseline_model_logging, outfile)
