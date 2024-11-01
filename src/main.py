import os
import yaml
import json

from data.ner_dataset import NER_Dataset
from data.wikiann_dataloader import WikiANN_Dataloader
from utils.logging import print_message, print_submessage

from torch.utils.data import DataLoader

# ------------------- EXAMPLE: Loading config.yaml file ------------------- #
def load_config(config_path="src/config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

config = load_config()

# ------------------- EXAMPLE: Loading Germanic languages high and low resource languages ------------------- #
print_message("Downloading data...")

wikiann_loader = WikiANN_Dataloader()

file_path = os.path.join(os.path.dirname(__file__), "../data/lang_groups.json")

with open(file_path, "r") as file:
    language_groups = json.load(file)

language_data = {}
for group_name, lang_dict in language_groups.items():

    print_submessage(f"Downloading data for {group_name} languages...")

    low_resource_langs = lang_dict["low_resource"]
    high_resource_langs = lang_dict["high_resource"]

    low_resource_train, low_resource_val, low_resource_test = wikiann_loader.load_split_data(low_resource_langs, shuffle=True)
    high_resource_data = wikiann_loader.load_data(high_resource_langs, shuffle=True)

    # Calculate how much of the high_resource language to keep based on the augumentation factor
    augumentation_factor = config["data"]["augumentation_factor"]
    high_resource_data = high_resource_data.head( len(low_resource_train) * augumentation_factor )

    language_data[group_name] = {
        "low_resource": {
            "train": low_resource_train,
            "val": low_resource_val,
            "test": low_resource_test
        },
        "high_resource": {
            "train": high_resource_data
        }
    }

# ------------------- EXAMPLE: Loading data into custom dataset and creating dataloader ------------------- #

print_message("Creating NER dataset...")

nerdataset = NER_Dataset(language_data["Germanic"]["low_resource"]["train"], config)
dataloader = DataLoader(nerdataset, batch_size=config["training"]["batch_size"], shuffle=True, collate_fn=nerdataset.collate_fn)