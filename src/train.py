import torch
from tqdm import tqdm
import torch.optim as optim
from utils.metrics import f1_score, prepare_labels

def train_model(model, train_dataloader, eval_dataloader, num_epochs, learning_rate, device):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_epoch(model, train_dataloader, optimizer, epoch, num_epochs, device)
        evaluate_epoch(model, eval_dataloader, device)

def train_epoch(model, dataloader, optimizer, epoch, num_epochs, device):
    model.train()
    total_loss = 0
    total_tokens = 0

    all_preds = torch.tensor([], dtype=torch.long, device=device)
    all_labels = torch.tensor([], dtype=torch.long, device=device)

    for idx, (input_ids, labels, attention_mask) in enumerate(pbar := tqdm(dataloader)):

        # Replace -100 in labels with 0 (or any valid tag) temporarily for CRF computation
        labels = torch.where(labels == model.label_pad_idx, torch.tensor(0).to(labels.device), labels)

        # Pass inputs through model
        optimizer.zero_grad()
        emissions = model(input_ids, attention_mask)
        
        # Calculate loss and per-token loss
        loss = model.loss(emissions, labels, attention_mask.bool())
        num_tokens = attention_mask.sum().item()
        per_token_loss = loss.item() / num_tokens if num_tokens > 0 else loss.item()

        # Prepare and colllect masked predictions and labels
        masked_predictions, masked_labels = prepare_labels(emissions, labels, attention_mask)
        all_preds = torch.cat((all_preds, masked_predictions), dim=0)
        all_labels = torch.cat((all_labels, masked_labels), dim=0)

        # Calculate F1-score and update optimizer
        avg_f1_score = f1_score(masked_predictions, masked_labels, model.num_tags)
        loss.backward()
        optimizer.step()

        # Update total loss and progress bar
        total_loss += loss.item()
        total_tokens += num_tokens
        pbar.set_description(f"Epoch {epoch + 1}/{num_epochs}     Per-token Loss: {per_token_loss:.3f}     F1-Score: {avg_f1_score:.3f}")

    # Calculate final loss and F1-score
    avg_token_loss = total_loss / total_tokens
    epoch_f1 = f1_score(all_preds, all_labels, model.num_tags)
    print(f"Epoch {epoch + 1}/{num_epochs}     Avg Token Loss: {avg_token_loss:.4f}     Avg F1-Score: {epoch_f1:.4f}")

def evaluate_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_tokens = 0

    all_preds = torch.tensor([], dtype=torch.long, device=device)
    all_labels = torch.tensor([], dtype=torch.long, device=device)

    with torch.no_grad():
        for idx, (input_ids, labels, attention_mask) in enumerate(pbar := tqdm(dataloader)):

            # Replace -100 in labels with 0 (or any valid tag) temporarily for CRF computation
            labels = torch.where(labels == model.label_pad_idx, torch.tensor(0).to(labels.device), labels)

            # Pass inputs through model
            emissions = model(input_ids, attention_mask)
            loss = model.loss(emissions, labels, attention_mask.bool())

            # Prepare and colllect masked predictions and labels
            masked_predictions, masked_labels = prepare_labels(emissions, labels, attention_mask)
            all_preds = torch.cat((all_preds, masked_predictions), dim=0)
            all_labels = torch.cat((all_labels, masked_labels), dim=0)

            # Calculate F1-score and update optimizer
            avg_f1_score = f1_score(masked_predictions, masked_labels, model.num_tags)

            # Update total loss, total tokens and progress bar
            total_loss += loss.item()
            total_tokens += attention_mask.sum().item()
            pbar.set_description(f"Evaluating:     F1-Score: {avg_f1_score:.3f}")

    # Calculate final loss and F1-score
    avg_token_loss = total_loss / total_tokens
    epoch_f1 = f1_score(all_preds, all_labels, model.num_tags)
    print(f"Evaluating:     Avg Token Loss: {avg_token_loss:.4f}     Avg F1-Score: {epoch_f1:.4f}")