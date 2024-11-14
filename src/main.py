from utils.logging import print_message
from utils.load_config import load_config

from data.ner_dataset import NER_Dataset
from data.NERDataset import NERDataset
from data.data_loader import LanguageDataLoader

from models.bert_bilstm_crf import BertBiLstmCRF
from models.BertModel import BERTBiLSTMCRF

from train import train
from train_mine import train_model

from torch.optim import AdamW
from transformers import BertConfig
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

# ------------------- Load config.yaml file ------------------- #

config = load_config()

# ------------------- Loading language groups containing high and low resource languages ------------------- #
print_message("Downloading data...")

dataloader = LanguageDataLoader(config)
language_data = dataloader.load_language_groups()

# ------------------- EXAMPLE: Loading data into custom dataset and creating dataloader ------------------- #

print_message("Creating NER dataset...")

nerdataset = NERDataset(language_data["Germanic"]["low_resource"]["train"], config)
dataloader = DataLoader(nerdataset, batch_size=config["training"]["batch_size"], shuffle=True, collate_fn=nerdataset.collate_fn)

print_message("Initializing model...")

model = BERTBiLSTMCRF(config["model"]["num_labels"])
model.to(config["model"]["device"])

print_message("Training...")
train_model(model, dataloader, config["training"]["epoch_num"], config["training"]["learning_rate"], config["model"]["device"])