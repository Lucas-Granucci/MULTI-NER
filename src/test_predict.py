import json
import pandas as pd
import torch
from tqdm import tqdm
from typing import Tuple, Dict, Any

from utils import set_seed
from config import ExperimentConfig
from models.BertBilstmCrf import BertBilstmCrf
from training.train_methods.pseudo_labeling import pseudo_labeling_train
from preprocessing.dataloader import LanguageDataManager

# Configuration for pseudo-labeling experiment
pseudo_labeling_config = ExperimentConfig(
    num_tags=7,
    batch_size=48,
    patience=5,
    epochs=20,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    bert_learning_rate=0.00003,
    lstm_learning_rate=0.005,
    crf_learning_rate=0.00005,
    seed=42,
    low_resource_base_count=240,  # 300 * 0.8
    results_dir="src/experiments/results/objective_IV",
    model_dir="src/models/pretrained",
    logging_dir="src/experiments/logging/objective_IV",
)


class PseudoLabelingExperiment:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.data_manager = LanguageDataManager()

    def run_experiment(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # Load language data
        low_resource_languages = ["mg", "fo", "co", "hsb", "bh", "cv"]
        language_data = self.data_manager.load_languages(low_resource_languages)

        results = {}

        for language, language_df in language_data.items():
            model_path = f"{self.config.model_dir}/{language}_pseudo_pretrained.pth"
            pseudo_model_path = f"{self.config.model_dir}/{language}_temp_pseudo.pth"

            # Load teacher model
            teacher_model = BertBilstmCrf(num_tags=self.config.num_tags)
            teacher_model.load_state_dict(
                torch.load(
                    f"{self.config.model_dir}/{language}_baseline_pretrained.pth",
                    weights_only=True,
                )
            )

            # Load pseudo-labeled data
            pseudo_labels_df = self.data_manager.load_data(f"data/pseudo_labels/{language}_pseudo_labels.csv")

            # Train and evaluate model with pseudo-labeling
            train_f1, test_f1 = pseudo_labeling_train(
                teacher_model,
                BertBilstmCrf,
                pseudo_labels_df,
                language_df,
                save_model=model_path,
                save_pseudo_model=pseudo_model_path,
                config=self.config,
            )

            print(
                f"Pseudo-labeling model on {language}:    Train F1: {train_f1:.4f}    Test F1: {test_f1:.4f}"
            )

            results[language] = {"train_f1": train_f1, "test_f1": test_f1}

        return results


def main():
    # Initialize experiment
    experiment = PseudoLabelingExperiment(pseudo_labeling_config)

    # Set random seed for reproducibility
    set_seed(pseudo_labeling_config.seed)

    # Run experiment
    results = experiment.run_experiment()

    # Save results to file
    output_path = f"{pseudo_labeling_config.results_dir}/3pseudo_labeling_model_performance.json"
    with open(output_path, "w") as outfile:
        json.dump(results, outfile)


if __name__ == "__main__":
    main()
