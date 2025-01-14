import json

from models.Bert import Bert
from models.BertBilstm import BertBilstm
from models.BertBilstmCrf import BertBilstmCrf
from models.XLMRoberta import XLMRoberta
from models.XLMRobertaBilstm import XLMRobertaBilstm
from models.XLMRobertaBilstmCrf import XLMRobertaBilstmCrf

from training.cv_train import cv_train
from training.train_configs import model_configs
from preprocessing.dataloader import load_data
from preprocessing.dataset import NERDataset

# Define languages
languages = {
    "fo": load_data("data/raw/fo_data.csv"),
    "co": load_data("data/raw/co_data.csv"),
    "hsb": load_data("data/raw/hsb_data.csv"),
    "bh": load_data("data/raw/bh_data.csv"),
    "cv": load_data("data/raw/cv_data.csv"),
    "mg": load_data("data/raw/mg_data.csv"),
}

# Define models
models = {
    "BERT": Bert,
    "BERT-Bilstm": BertBilstm,
    "BERT-Bilstm-CRF": BertBilstmCrf,
    "XLM-Roberta": XLMRoberta,
    "XLM-Roberta-Bilstm": XLMRobertaBilstm,
    "XLM-Roberta-Bilstm-CRF": XLMRobertaBilstmCrf,
}

model_performance = {}

# Iterate over every model and language
for model_name, model_type in models.items():
    for language, lang_df in languages.items():
        print(f"Testing model: {model_name} on language: {language}")

        # Create NER dataset for language
        dataset = NERDataset(
            texts=lang_df["tokens"].to_list(),
            tags=lang_df["ner_tags"].to_list(),
            tokenizer_type=model_type.get_model_type(),
        )

        # Train and evaluate model
        val_f1, train_f1 = cv_train(
            Model=model_type,
            dataset=dataset,
            k_splits=5,
            config=model_configs[model_name],
        )

        # Instantiate empty dict for model results
        model_performance.setdefault(model_name, {}).setdefault(language, {})

        # Store results
        model_performance[model_name][language] = {
            "train_f1": train_f1,
            "val_f1": val_f1,
        }

# Save results to json
with open("src/experiments/results/models_performance.json", "w") as outfile:
    json.dump(model_performance, outfile)
