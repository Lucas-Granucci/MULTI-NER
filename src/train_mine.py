import torch
import torch.optim as optim
from tqdm import tqdm

def train_model(model, dataloader, num_epochs, learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        total_batches = 0
        for idx, batch in enumerate(pbar := tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            batch_data, batch_token_starts, batch_labels = batch

            input_ids = batch_data
            attention_mask = batch_token_starts
            labels = batch_labels

            # Replace -100 in labels with 0 (or any valid tag) temporarily for CRF computation
            labels = torch.where(labels == model.label_pad_idx, torch.tensor(0).to(labels.device), labels)
            mask = labels != model.label_pad_idx

            optimizer.zero_grad()
            emissions = model(input_ids, attention_mask)
            
            # Calculate loss using masked labels and mask for padding
            loss = model.loss(emissions, labels, mask)

            # Calculate per-token loss
            num_tokens = mask.sum().item()
            per_token_loss = round(loss.item() / num_tokens, 3) if num_tokens > 0 else round(loss.item(), 3)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

            avg_loss = round(total_loss / total_batches, 3)
            pbar.set_description(f"Epoch {epoch + 1}/{num_epochs}     Per-token Loss: {per_token_loss}     Avg Loss: {avg_loss}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f}")
