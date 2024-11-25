from data.LangDataloader import LanguageDataLoader

from models.BertBilstmCrf import BERTBiLSTMCRF

from utils.logging import logger
from utils.load_config import load_config

# Load configuration
config = load_config()

NUM_LABELS = config["model"]["num_labels"]
DEVICE = config["model"]["device"]

# Load language data
logger.info("Downloading data...")
dataloader = LanguageDataLoader(config)
language_data = dataloader.load_language_groups()

# Define model
model = BERTBiLSTMCRF
