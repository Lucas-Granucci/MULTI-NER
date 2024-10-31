import os
import yaml
import json

from data.wikiann_dataloader import WikiANN_Dataloader
from data.ner_dataset import NER_Dataset

# ------------------- EXAMPLE: Loading Germanic languages high and low resource languages ------------------- #

if True:
    file_path = os.path.join(os.path.dirname(__file__), "../data/lang_groups.json")

    with open(file_path, "r") as file:
        language_groups = json.load(file)

    germanic_group = language_groups["Germanic"]
    low_resource_germanic = germanic_group["low_resource"]
    high_resource_germanic = germanic_group["high_resource"]

    wikiann_loader = WikiANN_Dataloader(low_resource_germanic)
    low_resource_germanic_dataset = wikiann_loader.load_data()

    #wikiann_loader = WikiANN_Dataloader(high_resource_germanic)
    #high_resource_germanic_dataset = wikiann_loader.load_data()


# ------------------- EXAMPLE: Loading config.yaml file ------------------- #
def load_config(config_path="src/config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

config = load_config()

nerdataset = NER_Dataset(low_resource_germanic_dataset, config)
print(len(nerdataset))
print(nerdataset[0])
print()
print(nerdataset[1])