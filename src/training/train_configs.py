import torch
from training.utils import TrainConfig

NUM_TAGS = 7
BATCH_SIZE = 16
NUM_EPOCHS = 15
MODEL_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_configs = {
    "BERT": TrainConfig(
        NUM_TAGS, BATCH_SIZE, 0.00002, None, None, NUM_EPOCHS, MODEL_DEVICE
    ),
    "BERT-Bilstm": TrainConfig(
        NUM_TAGS, BATCH_SIZE, 0.00002, 0.0005, None, NUM_EPOCHS, MODEL_DEVICE
    ),
    "BERT-Bilstm-CRF": TrainConfig(
        NUM_TAGS, BATCH_SIZE, 0.00003, 0.0005, 0.0005, NUM_EPOCHS, MODEL_DEVICE
    ),
    "XLM-Roberta": TrainConfig(
        NUM_TAGS, BATCH_SIZE, 0.00002, None, None, NUM_EPOCHS, MODEL_DEVICE
    ),
    "XLM-Roberta-Bilstm": TrainConfig(
        NUM_TAGS, BATCH_SIZE, 0.00002, 0.0005, None, NUM_EPOCHS, MODEL_DEVICE
    ),
    "XLM-Roberta-Bilstm-CRF": TrainConfig(
        NUM_TAGS, BATCH_SIZE, 0.00003, 0.0005, 0.0005, NUM_EPOCHS, MODEL_DEVICE
    ),
}
