import gc
import copy
import torch
import torch.optim as optim

from utils.metrics import calculate_f1

def train_model(model, optimizer, train_loader, val_loader, CONFIG):
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = "min", factor = 0.1, patience = 5)

    best_val_f1 = -float("inf")
    best_train_f1 = 0
    patience_counter = CONFIG.PATIENCE

    for _ in range(CONFIG.EPOCHS):
        _, train_f1 = train_epoch(model, train_loader, optimizer, CONFIG)
        val_loss, val_f1 = evaluate_epoch(model, val_loader, CONFIG)

        scheduler.step(val_loss)

        # Save state of best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_train_f1 = train_f1
            patience_counter = CONFIG.PATIENCE
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter -= 1

        if patience_counter == 0:
            break  # Stop training if model doesn't improve

    # Delete to clear up memory
    model.to("cpu")
    del optimizer, scheduler, model

    # Clear cache
    gc.collect()
    torch.cuda.empty_cache()

    return best_model_state, best_train_f1, best_val_f1


def train_epoch(model, dataloader, optimizer, CONFIG):
    model.train()
    total_loss, total_f1 = 0, 0

    for batch in dataloader:
        batch = {key : value.to(CONFIG.DEVICE) for key, value in batch.items()}

        optimizer.zero_grad()
        emissions, loss = model(**batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        pred_tags = model.decode(emissions, batch["attention_mask"])
        f1_score = calculate_f1(batch["target_tags"], pred_tags, batch["attention_mask"])
        total_f1 += f1_score

    return total_loss / len(dataloader), total_f1 / len(dataloader)

def evaluate_epoch(model, dataloader, CONFIG):
    model.eval()
    total_loss, total_f1 = 0, 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {key : value.to(CONFIG.DEVICE) for key, value in batch.items()}

            emissions, loss = model(**batch)
            total_loss += loss.item()

            pred_tags = model.decode(emissions, batch["attention_mask"])
            f1_score = calculate_f1(batch["target_tags"], pred_tags, batch["attention_mask"])
            total_f1 += f1_score

    return total_loss / len(dataloader), total_f1 / len(dataloader)