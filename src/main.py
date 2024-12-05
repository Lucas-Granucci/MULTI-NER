import torch
from data.LangDataloader import LanguageDataLoader

from data.NERDataloader import create_dataloaders
from models.XLMRobertaBilstmCrf import XLMRoBERTaBiLSTMCRF

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

model = XLMRoBERTaBiLSTMCRF(NUM_LABELS).to(DEVICE)
model.load_state_dict(torch.load("src/models/pretrained/best_germanic.pth", weights_only=True))

train_dataloader, test_dataloader, val_dataloader = create_dataloaders(language_data["germanic"], config, use_transfer_learning=False)

emissions, labels = predict_test(model, train_dataloader)
predictions = emissions.argmax(dim=-1)
print("SHAPE: ", emissions.shape)
print("PREDS SHAPE: ", predictions.shape)

id2label = config["data"]["id2label"]
batch_labels = [
    [id2label[idx.item()] for idx in sequence] 
    for sequence in predictions
]

actual_labels = [
    [id2label[idx.item()] for idx in sequence if idx != -100] 
    for sequence in labels
]

for thing, thing2 in zip(batch_labels, actual_labels):
    print("BATCH LABELS: ", thing)
    print("ACTUAL: ", thing2)
    print()