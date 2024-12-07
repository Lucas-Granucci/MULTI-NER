import torch
from tqdm import tqdm
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from utils.metrics import f1_score, prepare_labels


def train_model(model, train_dataloader, test_dataloader, model_dir, config):
    device = config["model"]["device"]
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
            _, train_f1 = train_epoch(model, train_dataloader, optimizer, device)
            epoch_f1 = evaluate_epoch(model, test_dataloader, device)

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


def evaluate_model(model, val_dataloader, config):
    device = device = config["model"]["device"]
    eval_f1 = evaluate_epoch(model, val_dataloader, device)

    return eval_f1


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_f1 = 0.0

    for input_ids, labels, attention_mask in dataloader:
        # Esnure mask ignores padding labels
        #adjusted_mask = attention_mask & (labels != model.label_pad_idx)

        # Forward pass
        optimizer.zero_grad(set_to_none=True)
        emissions = model(input_ids, attention_mask)

        # Calculate loss and backward pass
        loss = model.loss(emissions, labels, attention_mask.bool())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Prepare and colllect masked predictions and labels
        masked_predictions, masked_labels = prepare_labels(
            emissions, labels, attention_mask
        )
        batch_f1 = f1_score(masked_predictions, masked_labels, model.num_tags)
        total_f1 += batch_f1

        # Update total loss and progress bar
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    avg_f1 = total_f1 / len(dataloader)

    return avg_loss, avg_f1


def evaluate_epoch(model, dataloader, device):
    model.eval()
    total_f1 = 0.0
    
    dummy_label = model.num_tags
    dummy_tensor = torch.tensor(dummy_label, device=device)

    with torch.no_grad():
        for input_ids, labels, attention_mask in dataloader:
            # Replace -100 in labels with 0 (or any valid tag) temporarily for CRF computation
            # labels = torch.where(labels == model.label_pad_idx, dummy_tensor, labels)

            #adjusted_mask = attention_mask & (labels != -100)

            # Forward pass
            emissions = model(input_ids, attention_mask)

            # Compute batch F1
            masked_predictions, masked_labels = prepare_labels(
                emissions, labels, attention_mask
            )
            batch_f1 = f1_score(masked_predictions, masked_labels, model.num_tags)
            total_f1 += batch_f1

    avg_f1 = total_f1 / len(dataloader)

    return avg_f1


def predict_test(model, dataloader):
    for input_ids, _, attention_mask in dataloader:
        print(input_ids.shape)
        # Forward pass
        emissions = model(input_ids, attention_mask)
        print(emissions.shape)
        return emissions
