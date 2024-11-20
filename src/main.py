import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from train_eval import train_model, evaluate_model
from data.NERDataloader import create_dataloaders
from data.data_loader import LanguageDataLoader

from models.BertBilstmCrf import BERTBiLSTMCRF
from models.BertBilstm import BERTBiLSTM
from models.Bert import BERT
from models.XLMRobertaBilstmCrf import XLMRoBERTaBiLSTMCRF

from utils.logging import logger
from utils.load_config import load_config

# Load configuration
config = load_config()

NUM_LABELS = config["model"]["num_labels"]
DEVICE = config["model"]["device"]

# Load language data
logger.info("Downloading data...")
dataloader = LanguageDataLoader(config)
language_data = dataloader.load_language_groups()

# Helper function to train and evaluate model
def run_experiment(language_group, output_path, use_transfer_learning=False):
    logger.info(f"Setting up data for {'transfer learning' if use_transfer_learning else 'direct training'}...")
    train_dataloader, test_dataloader, val_dataloader = create_dataloaders(
        language_group, config, use_transfer_learning
    )

    logger.info("Initializing model...")
    #model = BERTBiLSTMCRF(NUM_LABELS).to(DEVICE)
    model = BERT(NUM_LABELS).to(DEVICE)

    logger.info("Starting training...")
    best_train_f1, best_epoch = train_model(model, train_dataloader, test_dataloader, output_path, config)
    logger.info(f"Training complete. Best Train F1-Score: {best_train_f1:.4f} attained at Epoch {best_epoch}")

    logger.info("Starting evaluation...")
    # TODO: This shouldnt be the model, it should input the path so it gets the best model
    eval_f1 = evaluate_model(model, val_dataloader, config)
    logger.info(f"Evaluation complete. Eval F1-Score: {eval_f1:.4f}")

# Run experiments
logger.info("Running experiment on Icelandic data...")
run_experiment(language_data["Germanic"], "src/models/pretrained/just_ice.pth", use_transfer_learning=False)

logger.info("Running experiment on Danish+Icelandic data...")
run_experiment(language_data["Germanic"], "src/models/pretrained/dan_ice.pth", use_transfer_learning=True)