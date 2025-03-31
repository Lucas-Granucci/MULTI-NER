from config import BaseConfig
from dataset import NERDataset
from torch.utils.data import DataLoader


def create_dataloader(lang_split_data, include_sentence=False):
    dataset = NERDataset(
        lang_split_data["tokens"].to_list(),
        lang_split_data["ner_tags"].to_list(),
        include_sentence=include_sentence,
    )
    return DataLoader(dataset, BaseConfig.BATCH_SIZE)


def create_dataloaders(lang_data):

    train_loader = create_dataloader(lang_data["train"])
    val_loader = create_dataloader(lang_data["val"])
    test_loader = create_dataloader(lang_data["test"])

    return train_loader, val_loader, test_loader
