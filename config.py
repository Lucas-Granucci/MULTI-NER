import torch

class BaseConfig:
    # Miscelanous
    RANDOM_STATE          = 42
    DEVICE                = torch.device("cuda")

    # Data
    low_resource_langs    = ["mg", "fo", "co", "hsb", "bh", "cv"]
    high_resource_langs   = ["id", "da", "it", "pl", "hi", "tr"]

    NUM_TAGS              = 7
    BATCH_SIZE            = 32
    MAX_SEQ_LEN           = 80

class TrainConfig(BaseConfig):
    EPOCHS                = 20
    PATIENCE              = 5
    BERT_LEARNING_RATE    = 0.00003
    LSTM_LEARNING_RATE    = 0.005
    CRF_LEARNING_RATE     = 0.00005
    WEIGHT_DECAY          = 0.02

class FineTuneConfig(BaseConfig):
    EPOCHS                = 15
    PATIENCE              = 3
    BERT_LEARNING_RATE    = 0.00002
    LSTM_LEARNING_RATE    = 0.003
    CRF_LEARNING_RATE     = 0.00003

class PseudoLabelingConfig(BaseConfig):
    EPOCHS                = 25
    PATIENCE              = 5
    BERT_LEARNING_RATE    = 0.00002
    LSTM_LEARNING_RATE    = 0.003
    CRF_LEARNING_RATE     = 0.00003

    CONFIDENCE_QUANTILE   = 0.965
    PSEUDO_DELAY          = 8
    ENTROPY_THRESHOLD     = 0.2