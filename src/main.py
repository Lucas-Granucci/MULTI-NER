import torch
from data.LangDataloader import LanguageDataLoader

from data.NERDataloader import create_dataloaders
from models.XLMRobertaBilstmCrf import XLMRoBERTaBiLSTMCRF
from models.Bert import BERT
from models.BertBilstmCrf import BERTBiLSTMCRF

from utils.logging import logger
from utils.load_config import load_config

from train_eval import predict_test

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

model = BERT(NUM_LABELS).to(DEVICE)
model.load_state_dict(torch.load("src/models/pretrained/best_germanic.pth", weights_only=True))

train_dataloader, test_dataloader, val_dataloader = create_dataloaders(language_data["germanic"], config, use_transfer_learning=False)

predictions, labels, sentence_lengths = predict_test(model, train_dataloader)

id2label = config["data"]["id2label"]
decoded_predictions = [
    [id2label[idx.item()] for idx in sequence] 
    for sequence in predictions
]

decoded_labels = [
    [id2label[idx.item()] for idx in sequence if idx != -100] 
    for sequence in labels
]

for prediction, label, sentence_length in zip(decoded_predictions, decoded_labels, sentence_lengths):
    print("PREDICTION: ", prediction[:sentence_length])
    print("LABEL:      ", label)
    print()