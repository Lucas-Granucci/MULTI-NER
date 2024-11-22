import gc
import torch

from utils.logging import logger
from train_eval import train_model, evaluate_model
from data.NERDataloader import create_dataloaders

# Helper function to train and evaluate model
def run_experiment(Model, language, language_data, output_path, config, use_transfer_learning=False):

    try:
        logger.info(f"Running experiment with {Model.__name__} on {language} data...")

        NUM_LABELS = config["model"]["num_labels"]
        DEVICE = config["model"]["device"]

        train_dataloader, test_dataloader, val_dataloader = create_dataloaders(
            language_data[language], config, use_transfer_learning
        )

        model = Model(NUM_LABELS).to(DEVICE)

        best_train_f1, best_epoch = train_model(model, train_dataloader, test_dataloader, output_path, config)
        logger.info(f"Training complete. Best Train F1-Score: {best_train_f1:.4f} attained at Epoch {best_epoch}")

        # Load the best model from training
        model.load_state_dict(torch.load(output_path, weights_only = True))
        eval_f1 = evaluate_model(model, val_dataloader, config)
        logger.info(f"Evaluation complete. Eval F1-Score: {eval_f1:.4f}")
    finally:
        del model
        del train_dataloader, test_dataloader, val_dataloader

        gc.collect()
        torch.cuda.empty_cache()

    return best_train_f1, eval_f1