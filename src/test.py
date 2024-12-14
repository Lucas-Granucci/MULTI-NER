from data.LangDataloader import LanguageDataLoader
from data.DataCreator import DataCreator
from utils.load_config import load_config

# Load configuration
config = load_config()

# Instantiate data creator
dataCreator = DataCreator()

dataloader = LanguageDataLoader(config)
language_data = dataloader.load_language_groups()


dataCreator.translate_text_data(
    language_data["germanic"]["high_resource"]["test"],
    "Frisan",
    "data/generated_data/frisian.txt",
)
