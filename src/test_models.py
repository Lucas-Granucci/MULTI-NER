import json
import torch
from typing import Tuple, Dict, Any

from utils import set_seed
from config import ExperimentConfig
from training.train_methods.train_evaluate import train_evaluate
from preprocessing.dataloader import LanguageDataManager

from models.Bert import Bert
from models.BertCrf import BertCrf
from models.BertBilstm import BertBilstm
from models.BertBilstmCrf import BertBilstmCrf

train_config = ExperimentConfig(
    # ------- Train params ------- #
    num_tags=7,
    batch_size=48,
    patience=5,
    epochs=20,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    # ------- Model params ------- #
    bert_learning_rate=0.00003,
    lstm_learning_rate=0.005,
    crf_learning_rate=0.00005,
    # ------- Other params ------- #
    seed=42,
    low_resource_base_count=240,  # 300 * 0.8
    results_dir="src/experiments/results/objective_II",
    model_dir="src/models/pretrained",
    logging_dir="src/experiments/logging/objective_II",
)


class ModelExperiment:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.data_manager = LanguageDataManager()
        self.models = {
            "BERT": Bert,
            "BERT-CRF": BertCrf,
            "BERT-Bilstm": BertBilstm,
            "BERT-Bilstm-CRF": BertBilstmCrf,
        }

    def run_experiment(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # Load language data
        low_resource_langs = ["mg", "fo", "co", "hsb", "bh", "cv"]
        language_data = self.data_manager.load_languages(low_resource_langs)

        results = {}
        logging = {}

        for model_name, ModelClass in self.models.items():
            results[model_name] = {}
            logging[model_name] = {}

            for lang, lang_df in language_data.items():
                model_path = f"{self.config.model_dir}/{lang}_{model_name}_pretrained.pth"

                print(f"Testing model: {model_name} on language: '{lang}'")

                # Train and evaluate model
                train_f1, test_f1, logging_results = train_evaluate(
                    model_class=ModelClass,
                    dataframe=lang_df,
                    config=self.config,
                    save_model=model_path,
                    verbose=True,
                )

                print(
                    f"{model_name} on {lang}:    Train F1: {train_f1:.4f}    Test F1: {test_f1:.4f}"
                )

                results[model_name][lang] = {"train_f1": train_f1, "test_f1": test_f1}
                logging[model_name][lang] = logging_results

        return results, logging


def main():
    config = train_config
    experiment = ModelExperiment(config)

    set_seed(config.seed)

    results, logging = experiment.run_experiment()

    # Save results
    output_path = f"{config.results_dir}/models_performance.json"
    with open(output_path, "w") as outfile:
        json.dump(results, outfile)


if __name__ == "__main__":
    main()
