import json
import torch

from training.train_methods.transfer_train import transfer_train
from models.BertBilstmCrf import BertBilstmCrf
from utils import set_seed
from preprocessing.dataloader import LanguageDataManager
from typing import Dict, Any
from tqdm import tqdm
from config import ExperimentConfig

# Configuration for the experiment
train_config = ExperimentConfig(
    num_tags=7,
    batch_size=48,
    patience=5,
    epochs=20,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    bert_learning_rate=0.00003,
    lstm_learning_rate=0.005,
    crf_learning_rate=0.0005,
    seed=42,
    low_resource_base_count=240,  # 300 * 0.8
    results_dir="src/experiments/results/objective_IV",
    model_dir="src/models/pretrained",
    logging_dir="src/experiments/logging/objective_IV",
)

class TransferLearningExperiment:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.data_manager = LanguageDataManager()

    def run_experiment(self, augmentation_factor: int) -> Dict[str, Any]:
        high_resource_count = self.config.low_resource_base_count * augmentation_factor

        # Load language data
        low_resource_langs = ["mg", "fo", "co", "hsb", "bh", "cv"]
        high_resource_langs = ["id", "is", "it", "pl", "hi", "tt"]

        low_resource_data, high_resource_data = self.data_manager.load_language_pairs(
            low_resource_langs,
            high_resource_langs,
            self.config.low_resource_base_count,
            high_resource_count,
        )

        results = {}
        for (low_lang, low_df), (high_lang, high_df) in zip(
            low_resource_data.items(), high_resource_data.items()
        ):
            model_path = f"{self.config.model_dir}/{low_lang}_{high_lang}_transfer_pretrained.pth"

            train_f1, test_f1, _ = transfer_train(
                model_class=BertBilstmCrf,
                low_resource_dataframe=low_df,
                high_resource_dataframe=high_df,
                config=self.config,
                save_model=model_path,
                verbose=False,
            )

            results[f"{low_lang}_{high_lang}"] = {
                "train_f1": train_f1,
                "test_f1": test_f1,
            }

        return results

def main():
    config = train_config
    experiment = TransferLearningExperiment(config)

    set_seed(config.seed)

    for augmentation_factor in tqdm(range(1, 13), desc="Training transfer learning"):
        results = experiment.run_experiment(augmentation_factor)

        # Save results
        output_path = f"{config.results_dir}/transfer_learning_performance_{augmentation_factor}.json"
        with open(output_path, "w") as outfile:
            json.dump(results, outfile)

if __name__ == "__main__":
    main()
