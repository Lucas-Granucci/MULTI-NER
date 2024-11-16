import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
from train import train_model
from data.NERDataset import NERDataset
from data.data_loader import LanguageDataLoader
from models.BertModel import BERTBiLSTMCRF

from utils.logging import logger
from utils.load_config import load_config

from torch.utils.data import DataLoader

# ------------------- Load config.yaml file ------------------- #

config = load_config()

# ------------------- Loading language groups containing high and low resource languages ------------------- #
logger.info("Downloading data...")

dataloader = LanguageDataLoader(config)
language_data = dataloader.load_language_groups()

# ------------------- EXAMPLE: Loading data into custom dataset and creating dataloader ------------------- #

logger.info("Creating Icelandic NER dataset...")

icelandic_train_df = language_data["Germanic"]["low_resource"]["train"]
icelandic_test_df = language_data["Germanic"]["low_resource"]["test"]

train_nerdataset = NERDataset(icelandic_train_df, config)
train_dataloader = DataLoader(train_nerdataset, batch_size=config["training"]["batch_size"], shuffle=True, collate_fn=train_nerdataset.collate_fn)

test_nerdataset = NERDataset(icelandic_test_df, config)
test_dataloader = DataLoader(test_nerdataset, batch_size=config["training"]["batch_size"], shuffle=True, collate_fn=test_nerdataset.collate_fn)

logger.info("Initializing model...")

model = BERTBiLSTMCRF(config["model"]["num_labels"])
model.to(config["model"]["device"])

logger.info("Training on Icelandic data...")
best_f1_score = train_model(model, train_dataloader, test_dataloader, "src/models/pretrained/just_ice.pth", config)
logger.info(f"Training complete. Best F1-Score: {best_f1_score:.4f}")

# ------------------- EXAMPLE: Loading data into custom dataset and creating dataloader ------------------- #

logger.info("Creating Danish+Icelandic NER dataset...")

danish_icelandic_train_df = pd.concat([language_data["Germanic"]["low_resource"]["train"], language_data["Germanic"]["high_resource"]["train"].head(10000)], ignore_index=True)
danish_icelandic_test_df = language_data["Germanic"]["low_resource"]["test"]

train_nerdataset = NERDataset(danish_icelandic_train_df, config)
train_dataloader = DataLoader(train_nerdataset, batch_size=config["training"]["batch_size"], shuffle=True, collate_fn=train_nerdataset.collate_fn)

test_nerdataset = NERDataset(danish_icelandic_test_df, config)
test_dataloader = DataLoader(test_nerdataset, batch_size=config["training"]["batch_size"], shuffle=True, collate_fn=test_nerdataset.collate_fn)

logger.info("Initializing model...")

model = BERTBiLSTMCRF(config["model"]["num_labels"])
model.to(config["model"]["device"])

logger.info("Training on Danish+Icelandic data...")
best_f1_score = train_model(model, train_dataloader, test_dataloader, "src/models/pretrained/dan_ice.pth", config)
logger.info(f"Training complete. Best F1-Score: {best_f1_score:.4f}")