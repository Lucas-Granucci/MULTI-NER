import torch
from tqdm import tqdm
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from utils.metrics import f1_score, prepare_labels


def train_model(model, train_dataloader, test_dataloader, model_dir, config):
    num_epochs = config["training"]["epoch_num"]
    f1_patience = config["training"]["f1_patience"]
    early_stopping = f1_patience

    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    scheduler = lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.3, total_iters=num_epochs
    )

    best_f1_score = -float("inf")
    best_epoch = -1

    with tqdm(total=num_epochs, desc="Training Progress", unit="epoch") as pbar:
        for epoch in range(num_epochs):
            # Training and evaluation
            _, train_f1 = train_epoch(model, train_dataloader, optimizer)
            epoch_f1 = evaluate_epoch(model, test_dataloader)

            # Save the best version of the model
            if epoch_f1 > best_f1_score:
                best_f1_score = epoch_f1
                best_epoch = epoch + 1
                torch.save(model.state_dict(), model_dir)

                early_stopping = f1_patience
            else:
                early_stopping -= 1

            # Update progress bar
            pbar.set_postfix(
                train_f1=f"{train_f1:.4f}",
                val_f1=f"{epoch_f1:.4f}",
                best_f1=f"{best_f1_score:.4f}",
            )
            pbar.update(1)

            # Step scheduler
            scheduler.step

            # Early stopping
            if early_stopping == 0:
                break

    return best_f1_score, best_epoch


def evaluate_model(model, val_dataloader):
    eval_f1 = evaluate_epoch(model, val_dataloader)
    return eval_f1


def train_epoch(model, dataloader, optimizer):
    model.train()
    total_loss = 0.0
    total_f1 = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]

        # Forward pass
        optimizer.zero_grad(set_to_none=True)
        emissions = model(input_ids, attention_mask)

        # Calculate loss and backward pass
        loss = model.loss(emissions, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Prepare and colllect masked predictions and labels
        decoded_emissions = model.decode(emissions)
        masked_predictions, masked_labels = prepare_labels(
            decoded_emissions, labels, attention_mask
        )
        batch_f1 = f1_score(masked_predictions, masked_labels, model.num_tags)
        total_f1 += batch_f1

        # Update total loss and progress bar
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    avg_f1 = total_f1 / len(dataloader)

    return avg_loss, avg_f1


def evaluate_epoch(model, dataloader):
    model.eval()
    total_f1 = 0.0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            attention_mask = batch["attention_mask"]

            # Forward pass
            emissions = model(input_ids, attention_mask)

            # Compute batch F1
            decoded_emissions = model.decode(emissions)
            masked_predictions, masked_labels = prepare_labels(
                decoded_emissions, labels, attention_mask
            )
            batch_f1 = f1_score(masked_predictions, masked_labels, model.num_tags)
            total_f1 += batch_f1

    avg_f1 = total_f1 / len(dataloader)

    return avg_f1


def predict_test(model, dataloader):
    for batch in dataloader:

        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]
        sentences = batch["sentences"]

        # Forward pass
        emissions = model(input_ids, attention_mask)
        decoded_emissions = model.decode(emissions)

        sentence_lengths = [len(sentence[1]) for sentence in sentences]
        return decoded_emissions, labels, sentence_lengths
