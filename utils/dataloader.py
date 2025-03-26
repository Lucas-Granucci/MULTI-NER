from utils.dataset import NERDataset
from torch.utils.data import DataLoader

def create_dataloader(lang_split_data, CONFIG, include_sentence=False):
    dataset = NERDataset(
        lang_split_data["tokens"].to_list(),
        lang_split_data["ner_tags"].to_list(),
        include_sentence = include_sentence
    )
    return DataLoader(dataset, CONFIG.BATCH_SIZE)

def create_dataloaders(lang_data, CONFIG):

    train_loader = create_dataloader(lang_data["train"], CONFIG)
    val_loader = create_dataloader(lang_data["val"], CONFIG)
    test_loader = create_dataloader(lang_data["test"], CONFIG)

    return train_loader, val_loader, test_loader