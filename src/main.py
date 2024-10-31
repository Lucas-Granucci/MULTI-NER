import os
import json
from data.wikiann_dataloader import WikiANN_Dataloader

# ------------------- EXAMPLE: Loading Germanic languages high and low resource languages ------------------- #
file_path = os.path.join(os.path.dirname(__file__), "../data/lang_groups.json")

with open(file_path, "r") as file:
    language_groups = json.load(file)

germanic_group = language_groups["Germanic"]
low_resource_germanic = germanic_group["low_resource_languages"]
high_resource_germanic = germanic_group["high_resource_languages"]

wikiann_loader = WikiANN_Dataloader(low_resource_germanic)
low_resource_germanic_dataset = wikiann_loader.load_data()

wikiann_loader = WikiANN_Dataloader(high_resource_germanic)
high_resource_germanic_dataset = wikiann_loader.load_data()
