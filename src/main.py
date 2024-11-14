import os

from train import train_model
from data.NERDataset import NERDataset
from data.data_loader import LanguageDataLoader
from models.BertModel import BERTBiLSTMCRF

from utils.logging import print_message
from utils.load_config import load_config

from torch.utils.data import DataLoader

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ------------------- Load config.yaml file ------------------- #

config = load_config()

# ------------------- Loading language groups containing high and low resource languages ------------------- #
print_message("Downloading data...")

dataloader = LanguageDataLoader(config)
language_data = dataloader.load_language_groups()

# ------------------- EXAMPLE: Loading data into custom dataset and creating dataloader ------------------- #

print_message("Creating NER dataset...")

train_nerdataset = NERDataset(language_data["Germanic"]["low_resource"]["train"], config)
train_dataloader = DataLoader(train_nerdataset, batch_size=config["training"]["batch_size"], shuffle=True, collate_fn=train_nerdataset.collate_fn)

eval_nerdataset = NERDataset(language_data["Germanic"]["low_resource"]["test"], config)
eval_dataloader = DataLoader(eval_nerdataset, batch_size=config["training"]["batch_size"], shuffle=True, collate_fn=eval_nerdataset.collate_fn)

print_message("Initializing model...")

model = BERTBiLSTMCRF(config["model"]["num_labels"])
model.to(config["model"]["device"])

print_message("Training...")
train_model(model, train_dataloader, eval_dataloader, config["training"]["epoch_num"], config["training"]["learning_rate"], config["model"]["device"])