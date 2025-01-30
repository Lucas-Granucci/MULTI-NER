import json
from training.cv_transfer_train import cv_transfer_train
from training.train_configs import model_configs
from preprocessing.dataloader import load_data
from models.BertCrf import BertCrf

LOW_RESOURCE_COUNT = 300
AUGUMENTATION_FACTOR = 1

HIGH_RESOURCE_COUNT = LOW_RESOURCE_COUNT * AUGUMENTATION_FACTOR

# Define languages
low_resource_languages = {
    "fo": load_data("data/labeled/fo_data.csv").head(LOW_RESOURCE_COUNT),
}

high_resource_languages = {
    "is": load_data("data/labeled/is_data.csv").head(HIGH_RESOURCE_COUNT)
}

# Define model
model_name = "BERT-CRF"
baseline_model = BertCrf

transfer_learning_performance = {}

# Iterate over languages
for low_resource_batch, high_resource_batch in zip(
    low_resource_languages.items(), high_resource_languages.items()
):
    # Extract language datat
    low_resource_lang, low_resource_lang_df = low_resource_batch
    high_resource_lang, high_resource_lang_df = high_resource_batch

    print(
        f"Testing: {model_name} on language: '{low_resource_lang}' with '{high_resource_lang}' transfer learning"
    )

    # Train and evaluate model
    val_f1, train_f1 = cv_transfer_train(
        Model=baseline_model,
        low_resource_dataframe=low_resource_lang_df,
        high_resource_dataframe=high_resource_lang_df,
        k_splits=5,
        config=model_configs[model_name],
    )

    # Save model results
    transfer_learning_performance[f"{low_resource_lang}_{high_resource_lang}"] = {
        "train_f1": train_f1,
        "val_f1": val_f1,
    }

    print(
        f"{model_name} on {low_resource_lang}_{high_resource_lang} -- Val F1: {val_f1}, Train F1: {train_f1}"
    )

# Save results to json
with open("src/experiments/results/transfer_learning_performance.json", "w") as outfile:
    json.dump(transfer_learning_performance, outfile)
