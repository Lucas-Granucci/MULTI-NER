import json
from data.LangDataloader import LanguageDataLoader
from models.BertBilstmCrf import BERTBiLSTMCRF

from utils.logging import logger
from utils.load_config import load_config
from utils.run_experiment import run_experiment


# Load configuration
config = load_config()

NUM_LABELS = config["model"]["num_labels"]
DEVICE = config["model"]["device"]

# Load language data
logger.info("Downloading data...")
dataloader = LanguageDataLoader(config)
language_data = dataloader.load_language_groups()

# Define model
NUM_LABELS = config["model"]["num_labels"]
DEVICE = config["model"]["device"]

model = BERTBiLSTMCRF(NUM_LABELS).to(DEVICE)

# Train and evaluate model
model_performance_results = {}
for language in language_data:
    train_f1, eval_f1 = run_experiment(
        model,
        language,
        language_data,
        f"src/models/pretrained/bertbilstmcrf_{language}.pth",
        config,
        use_transfer_learning=False,
    )

    # Initialize nested dictionaries
    model_performance_results.setdefault(model.__name__, {}).setdefault(language, {})

    model_performance_results[model.__name__][language]["train_f1"] = train_f1
    model_performance_results[model.__name__][language]["eval_f1"] = eval_f1

# Save model performance results to json
with open("results/baseline/model_performance.json", "w") as outfile:
    json.dump(model_performance_results, outfile)
