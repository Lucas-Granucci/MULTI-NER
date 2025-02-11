import json
from training.transfer_train import transfer_train
from training.train_configs import model_configs
from preprocessing.dataloader import load_data
from models.BertBilstmCrf import BertBilstmCrf
from utils import set_seed

set_seed(42)

for i in range(6, 13):
    LOW_RESOURCE_COUNT = int(300 * 0.8)
    AUGUMENTATION_FACTOR = i

    HIGH_RESOURCE_COUNT = LOW_RESOURCE_COUNT * AUGUMENTATION_FACTOR

    # Define languages
    low_resource_languages = {
        "mg": load_data("data/labeled/mg_data.csv").head(LOW_RESOURCE_COUNT),
        "fo": load_data("data/labeled/fo_data.csv").head(LOW_RESOURCE_COUNT),
        "co": load_data("data/labeled/co_data.csv").head(LOW_RESOURCE_COUNT),
        "hsb": load_data("data/labeled/hsb_data.csv").head(LOW_RESOURCE_COUNT),
        "bh": load_data("data/labeled/bh_data.csv").head(LOW_RESOURCE_COUNT),
        "cv": load_data("data/labeled/cv_data.csv").head(LOW_RESOURCE_COUNT),
    }

    high_resource_languages = {
        "id": load_data("data/labeled/id_data.csv").head(HIGH_RESOURCE_COUNT),
        "is": load_data("data/labeled/is_data.csv").head(HIGH_RESOURCE_COUNT),
        "it": load_data("data/labeled/it_data.csv").head(HIGH_RESOURCE_COUNT),
        "pl": load_data("data/labeled/pl_data.csv").head(HIGH_RESOURCE_COUNT),
        "hi": load_data("data/labeled/hi_data.csv").head(HIGH_RESOURCE_COUNT),
        "tt": load_data("data/labeled/tt_data.csv").head(HIGH_RESOURCE_COUNT),
    }

    # Define model
    model_name = "BERT-Bilstm-CRF"
    model_type = BertBilstmCrf

    transfer_learning_performance = {}

    # Iterate over languages
    for low_resource_batch, high_resource_batch in zip(
        low_resource_languages.items(), high_resource_languages.items()
    ):
        # Extract language datat
        low_resource_lang, low_resource_lang_df = low_resource_batch
        high_resource_lang, high_resource_lang_df = high_resource_batch

        print(
            f"""Testing: {model_name} with transfer learning
            Low-resource language: {low_resource_lang}    High-resource language: {high_resource_lang}    Augumentation factor: {AUGUMENTATION_FACTOR}    
            """
        )

        # Train and evaluate model
        train_f1, test_f1, logging_results = transfer_train(
            ModelClass=model_type,
            low_resource_dataframe=low_resource_lang_df,
            high_resource_dataframe=high_resource_lang_df,
            config=model_configs[model_name],
            save_model=f"src/models/pretrained/{low_resource_lang}_{high_resource_lang}_transfer_pretrained.pth",
            verbose=False,
        )

        # Save model results
        transfer_learning_performance[f"{low_resource_lang}_{high_resource_lang}"] = {
            "train_f1": train_f1,
            "test_f1": test_f1,
        }

        print(
            f"{model_name} on {low_resource_lang}_{high_resource_lang} -- Test F1: {test_f1}, Train F1: {train_f1}"
        )

    # Save results to json
    with open(
        f"src/experiments/results/objective_IV/new_transfer_learning_performance{AUGUMENTATION_FACTOR}.json",
        "w",
    ) as outfile:
        json.dump(transfer_learning_performance, outfile)
