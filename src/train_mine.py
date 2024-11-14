import torch
import torch.optim as optim
from utils.metrics import f1_score
from tqdm import tqdm

def train_model(model, dataloader, num_epochs, learning_rate, device):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        total_batches = 0

        all_preds = torch.tensor([], dtype=torch.long, device=device)
        all_labels = torch.tensor([], dtype=torch.long, device=device)

        for idx, batch in enumerate(pbar := tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            batch_data, batch_token_starts, batch_labels = batch

            input_ids = batch_data
            attention_mask = batch_token_starts
            labels = batch_labels

            # Replace -100 in labels with 0 (or any valid tag) temporarily for CRF computation
            labels = torch.where(labels == model.label_pad_idx, torch.tensor(0).to(labels.device), labels)

            optimizer.zero_grad()
            emissions = model(input_ids, attention_mask)
            
            # Calculate loss using masked labels and mask for padding
            loss = model.loss(emissions, labels, attention_mask.bool())

            # Calculate per-token loss
            num_tokens = attention_mask.sum().item()
            per_token_loss = round(loss.item() / num_tokens, 3) if num_tokens > 0 else round(loss.item(), 3)

            # Prepare outputs for F1-scoring
            predicted_labels = emissions.argmax(dim=-1)

            # Flatten for direct comparison
            predicted_labels = torch.flatten(predicted_labels)
            labels = torch.flatten(labels)
            attention_mask = torch.flatten(attention_mask)

            mask = attention_mask == 1

            masked_predictions = predicted_labels[mask]
            masked_labels = labels[mask]

            all_preds = torch.cat((all_preds, masked_predictions), dim=0)
            all_labels = torch.cat((all_labels, masked_labels), dim=0)

            # Calculate current batch's F1-score
            avg_f1_score = round(f1_score(masked_predictions, masked_labels, model.num_tags), 3)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

            avg_loss = round(total_loss / total_batches, 3)
            pbar.set_description(f"Epoch {epoch + 1}/{num_epochs}     Per-token Loss: {per_token_loss}     Avg Loss: {avg_loss}     F1-Score: {avg_f1_score}")

        avg_loss = total_loss / len(dataloader)
        epoch_f1 = f1_score(all_preds, all_labels, model.num_tags)
        print(f"Epoch {epoch + 1}/{num_epochs}     Avg Loss: {avg_loss:.4f}     Avg F1-Score: {epoch_f1:.4f}")
