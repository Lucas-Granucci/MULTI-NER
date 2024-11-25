import json

from data.LangDataloader import LanguageDataLoader

from models.Bert import BERT
from models.BertBilstm import BERTBiLSTM
from models.BertBilstmCrf import BERTBiLSTMCRF

from models.XLMRoberta import XLMRoBERTa
from models.XLMRobertaBilstm import XLMRoBERTaBiLSTM
from models.XLMRobertaBilstmCrf import XLMRoBERTaBiLSTMCRF

from utils.logging import logger
from utils.load_config import load_config
from utils.run_experiment import run_experiment

# Load configuration
config = load_config()

# Load language data
logger.info("Downloading data...")
dataloader = LanguageDataLoader(config)
language_data = dataloader.load_language_groups()

# Define models
models = [
    BERT,
    BERTBiLSTM,
    BERTBiLSTMCRF,
    XLMRoBERTa,
    XLMRoBERTaBiLSTM,
    XLMRoBERTaBiLSTMCRF,
]
languages = ["english", "arabic", "spanish", "russian", "turkish"]

# Run experiments
model_performance_results = {}

for model in models:

    for language in languages:

        train_f1, eval_f1 = run_experiment(
            model,
            language,
            language_data,
            f"src/models/pretrained/best_{language}.pth",
            config,
            use_transfer_learning=False,
        )

        # Initialize nested dictionaries
        model_performance_results.setdefault(model.__name__, {}).setdefault(
            language, {}
        )

        model_performance_results[model.__name__][language]["train_f1"] = train_f1
        model_performance_results[model.__name__][language]["eval_f1"] = eval_f1

# Save model performance results to json
with open("results/model_performance.json", "w") as outfile:
    json.dump(model_performance_results, outfile)
