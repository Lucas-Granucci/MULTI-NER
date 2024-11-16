import torch
from rich.progress import Progress
from rich.console import Console
import torch.optim as optim
from utils.metrics import f1_score, prepare_labels

console = Console()

def train_model(model, train_dataloader, test_dataloader, model_dir, config):

    device = config["model"]["device"]
    num_epochs = config["training"]["epoch_num"]

    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    best_f1_score = -float('inf')

    for epoch in range(num_epochs):
        train_epoch(model, train_dataloader, optimizer, epoch, num_epochs, device)
        epoch_f1 = evaluate_epoch(model, test_dataloader, device)

        console.rule()

        if epoch_f1 > best_f1_score:
            best_f1_score = epoch_f1
            torch.save(model.state_dict(), model_dir)

    return best_f1_score

def train_epoch(model, dataloader, optimizer, epoch, num_epochs, device):
    model.train()
    total_loss = 0
    total_tokens = 0

    all_preds = torch.tensor([], dtype=torch.long, device=device)
    all_labels = torch.tensor([], dtype=torch.long, device=device)

    with Progress() as progress:
        task = progress.add_task(f"Epoch {epoch + 1}/{num_epochs}", total=len(dataloader))
        for (input_ids, labels, attention_mask) in dataloader:

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
            progress.update(
                task,
                advance=1,
                description=(
                    f"[bold blue]Epoch {epoch + 1}/{num_epochs}[/bold blue] "
                    f"[red]Loss:[/red] {per_token_loss:.3f} "
                    f"[green]F1-Score:[/green] {avg_f1_score:.3f}"
                ),
            )

        # Calculate final loss and F1-score
        avg_token_loss = total_loss / total_tokens
        epoch_f1 = f1_score(all_preds, all_labels, model.num_tags)
        progress.update(
            task,
            completed=len(dataloader),
            description=(
                f"[bold green]Completed Epoch {epoch + 1}/{num_epochs}[/bold green] "
                f"[red]Avg Loss:[/red] {avg_token_loss:.4f} "
                f"[green]Avg F1-Score:[/green] {epoch_f1:.4f}"
            )
        )

def evaluate_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_tokens = 0

    all_preds = torch.tensor([], dtype=torch.long, device=device)
    all_labels = torch.tensor([], dtype=torch.long, device=device)

    with Progress() as progress:
        task = progress.add_task("Evaluating", total=len(dataloader))
        for (input_ids, labels, attention_mask) in dataloader:

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
            progress.update(
                task,
                advance=1,
                description=(
                    f"[bold yellow]Evaluating[/bold yellow]: "
                    f"[green]F1-Score:[/green] {avg_f1_score:.3f}"
                )
            )

        # Calculate final loss and F1-score
        avg_token_loss = total_loss / total_tokens
        epoch_f1 = f1_score(all_preds, all_labels, model.num_tags)

        progress.update(
            task,
            description=(
                f"[bold green]Evaluation Complete[/bold green]: "
                f"[red]Avg Loss:[/red] {avg_token_loss:.4f} "
                f"[green]Avg F1-Score:[/green] {epoch_f1:.4f}"
            )
        )

    return epoch_f1