import json

from models.Bert import Bert
from models.BertCrf import BertCrf
from models.BertBilstm import BertBilstm
from models.BertBilstmCrf import BertBilstmCrf

from training.train_evaluate import train_evaluate
from training.train_configs import model_configs
from preprocessing.dataloader import load_data
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

# Define models
models = {
    "BERT": Bert,
    "BERT-CRF": BertCrf,
    "BERT-Bilstm": BertBilstm,
    "BERT-Bilstm-CRF": BertBilstmCrf,
}

model_performance = {}

# Iterate over every model and language
for model_name, model_type in models.items():
    for language, lang_df in languages.items():
        print(f"Testing model: {model_name} on language: '{language}'")

        # Train and evaluate model
        train_f1, test_f1, logging_results = train_evaluate(
            ModelClass=model_type,
            dataframe=lang_df,
            config=model_configs[model_name],
            save_model=f"src/models/pretrained/{language}_testing_pretrained.pth",
            verbose=True,
        )

        # Instantiate empty dict for model results
        model_performance.setdefault(model_name, {}).setdefault(language, {})

        # Store results
        model_performance[model_name][language] = {
            "train_f1": train_f1,
            "test_f1": test_f1,
        }

        print(f"{model_name} on {language} -- Test F1: {test_f1}, Train F1: {train_f1}")

# Save results to json
with open(
    "src/experiments/results/objective_II/models_performance.json", "w"
) as outfile:
    json.dump(model_performance, outfile)
