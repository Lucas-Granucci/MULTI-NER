import torch
from training.utils import TrainConfig

NUM_TAGS = 7
BATCH_SIZE = 16
PATIENCE = 5
EPOCHS = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_configs = {
    "BERT": TrainConfig(
        NUM_TAGS, BATCH_SIZE, 0.00002, None, None, PATIENCE, EPOCHS, DEVICE
    ),
    "BERT-Bilstm": TrainConfig(
        NUM_TAGS, BATCH_SIZE, 0.00002, 0.0005, None, PATIENCE, EPOCHS, DEVICE
    ),
    "BERT-Bilstm-CRF": TrainConfig(
        NUM_TAGS, BATCH_SIZE, 0.00003, 0.0005, 0.0005, PATIENCE, EPOCHS, DEVICE
    ),
    "XLM-Roberta": TrainConfig(
        NUM_TAGS, BATCH_SIZE, 0.00002, None, None, PATIENCE, EPOCHS, DEVICE
    ),
    "XLM-Roberta-Bilstm": TrainConfig(
        NUM_TAGS, BATCH_SIZE, 0.00002, 0.0005, None, PATIENCE, EPOCHS, DEVICE
    ),
    "XLM-Roberta-Bilstm-CRF": TrainConfig(
        NUM_TAGS,
        BATCH_SIZE,
        0.00003,
        0.0005,
        0.0005,
        PATIENCE,
        EPOCHS,
        DEVICE,
    ),
}
