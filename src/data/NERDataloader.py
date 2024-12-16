import pandas as pd
from data.NERDataset import NERDataset
from torch.utils.data import DataLoader


def create_dataloaders(
    language_data, tokenizer_type, config, use_transfer_learning=False
):
    low_resource_train = language_data["low_resource"]["train"]
    low_resource_test = language_data["low_resource"]["test"]
    low_resource_val = language_data["low_resource"]["val"]

    high_resource_train = language_data["high_resource"]["train"]

    # Calculate amount of high resource data to include in training data
    if use_transfer_learning:
        augumentation_factor = config["data"]["augumentation_factor"]
        augumentation_amount = len(low_resource_train) * augumentation_factor
        high_resource_train = high_resource_train.head(augumentation_amount)

        complete_train = pd.concat([high_resource_train, low_resource_train])
    else:
        complete_train = low_resource_train

    train_nerdataset = NERDataset(complete_train, tokenizer_type, config)
    train_dataloader = DataLoader(
        train_nerdataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=train_nerdataset.collate_fn,
    )

    test_nerdataset = NERDataset(low_resource_test, tokenizer_type, config)
    test_dataloader = DataLoader(
        test_nerdataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=test_nerdataset.collate_fn,
    )

    val_nerdataset = NERDataset(low_resource_val, tokenizer_type, config)
    val_dataloader = DataLoader(
        val_nerdataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=val_nerdataset.collate_fn,
    )

    return train_dataloader, test_dataloader, val_dataloader
