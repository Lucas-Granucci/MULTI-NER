import torch
from training.train_utils import TrainConfig

NUM_TAGS = 7  # Number of unqique NER tags
BATCH_SIZE = 48  # Dataloader batches
PATIENCE = 5  # Early-stopping patience
EPOCHS = 20  # Number of epochs for training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_configs = {
    "BERT": TrainConfig(
        NUM_TAGS,
        BATCH_SIZE,
        3e-5,  # BERT LR
        None,
        None,
        PATIENCE,
        EPOCHS,
        DEVICE,
    ),
    "BERT-CRF": TrainConfig(
        NUM_TAGS,
        BATCH_SIZE,
        3e-5,  # BERT LR
        None,
        5e-5,  # CRF LR
        PATIENCE,
        EPOCHS,
        DEVICE,
    ),
    "BERT-Bilstm": TrainConfig(
        NUM_TAGS,
        BATCH_SIZE,
        3e-5,  # BERT LR
        5e-3,  # LSTM LR
        None,
        PATIENCE,
        EPOCHS,
        DEVICE,
    ),
    "BERT-Bilstm-CRF": TrainConfig(
        NUM_TAGS,
        BATCH_SIZE,
        3e-5,  # BERT LR
        5e-3,  # LSTM LR
        5e-5,  # CRF LR
        PATIENCE,
        EPOCHS,
        DEVICE,
    ),
}
