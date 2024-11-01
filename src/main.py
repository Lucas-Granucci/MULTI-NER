import yaml

from utils.logging import print_message
from utils.load_config import load_config

from data.ner_dataset import NER_Dataset
from data.data_loader import LanguageDataLoader

from torch.utils.data import DataLoader

# ------------------- Load config.yaml file ------------------- #

config = load_config()

# ------------------- Loading language groups containing high and low resource languages ------------------- #
print_message("Downloading data...")

dataloader = LanguageDataLoader(config)
language_data = dataloader.load_language_groups()

# ------------------- EXAMPLE: Loading data into custom dataset and creating dataloader ------------------- #

print_message("Creating NER dataset...")

nerdataset = NER_Dataset(language_data["Germanic"]["low_resource"]["train"], config)
dataloader = DataLoader(nerdataset, batch_size=config["training"]["batch_size"], shuffle=True, collate_fn=nerdataset.collate_fn)