import torch
import time
from rich.progress import Progress
from rich.console import Console
import torch.optim as optim
from utils.metrics import f1_score, prepare_labels
from numpy import mean

console = Console()

def train_model(model, train_dataloader, test_dataloader, model_dir, config):

    device = config["model"]["device"]
    num_epochs = config["training"]["epoch_num"]

    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    best_f1_score = -float('inf')
    best_epoch = -1

    for epoch in range(num_epochs):
        train_epoch(model, train_dataloader, optimizer, epoch, num_epochs, device)
        epoch_f1 = evaluate_epoch(model, test_dataloader, device)

        console.rule()

        if epoch_f1 > best_f1_score:
            best_f1_score = epoch_f1
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_dir)

    return best_f1_score, best_epoch

def evaluate_model(model, val_dataloader, config):

    device = device = config["model"]["device"]
    eval_f1 = evaluate_epoch(model, val_dataloader, device)

    return eval_f1

def train_epoch(model, dataloader, optimizer, epoch, num_epochs, device):
    model.train()
    total_loss = 0

    all_preds, all_labels = [], []

    zero_tensor = torch.tensor(0, device=device)

    with Progress() as progress:
        task = progress.add_task(f"Epoch {epoch + 1}/{num_epochs}", total=len(dataloader))
        for batch_idx, (input_ids, labels, attention_mask) in enumerate(dataloader):

            # Replace -100 in labels with 0 (or any valid tag) temporarily for CRF computation
            labels = torch.where(labels == model.label_pad_idx, zero_tensor, labels)

            # Pass inputs through model
            optimizer.zero_grad()
            emissions = model(input_ids, attention_mask)
            
            # Calculate loss and update weights
            loss = model.loss(emissions, labels, attention_mask.bool())
            loss.backward()
            optimizer.step()

            # Prepare and colllect masked predictions and labels
            masked_predictions, masked_labels = prepare_labels(emissions, labels, attention_mask)
            all_preds.append(masked_predictions)
            all_labels.append(masked_labels)

            # Update total loss and progress bar
            total_loss += loss.item()

            if batch_idx % 10 == 0:
                progress.update(
                    task,
                    advance=10,
                    description=(
                        f"[bold blue]Epoch {epoch + 1}/{num_epochs}[/bold blue] "
                    ),
                )

        # Concatenate predictions and labels once
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Calculate metrics
        epoch_f1 = f1_score(all_preds, all_labels, model.num_tags)
        avg_loss = total_loss / len(dataloader)

        # Update progress bar
        progress.update(
            task,
            completed=len(dataloader),
            description=(
                f"[bold green]Completed Epoch {epoch + 1}/{num_epochs}[/bold green] "
                f"[red]Avg Loss:[/red] {avg_loss:.4f} "
                f"[green]Avg F1-Score:[/green] {epoch_f1:.4f}"
            )
        )

def evaluate_epoch(model, dataloader, device):
    model.eval()

    all_preds, all_labels = [], []
    zero_tensor = torch.tensor(0, device=device)

    with torch.no_grad():
        with Progress() as progress:
            task = progress.add_task("Evaluating", total=len(dataloader))
            for batch_idx, (input_ids, labels, attention_mask) in enumerate(dataloader):

                # Replace -100 in labels with 0 (or any valid tag) temporarily for CRF computation
                labels = torch.where(labels == model.label_pad_idx, zero_tensor, labels)

                # Pass inputs through model
                emissions = model(input_ids, attention_mask)

                # Prepare and colllect masked predictions and labels
                masked_predictions, masked_labels = prepare_labels(emissions, labels, attention_mask)
                all_preds.append(masked_predictions)
                all_labels.append(masked_labels)

                # Update progress bar
                if batch_idx % 10 == 0:
                    progress.update(
                        task,
                        advance=10,
                        description=(
                            f"[bold yellow]Evaluating[/bold yellow]: "
                        )
                    )

            # Concatenate predictions and labels once
            all_preds = torch.cat(all_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            # Calculate metrics
            epoch_f1 = f1_score(all_preds, all_labels, model.num_tags)

            progress.update(
                task,
                description=(
                    f"[bold green]Evaluation Complete[/bold green]: "
                    f"[green]Avg F1-Score:[/green] {epoch_f1:.4f}"
                )
            )

    return epoch_f1