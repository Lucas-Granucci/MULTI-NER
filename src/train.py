import torch
import torch.nn as nn
from tqdm import tqdm
from utils.metrics import f1_score

def train_epoch(train_loader, model, optimizer, scheduler, epoch, config):
    model.train()

    train_losses = 0
    for idx, batch_samples in enumerate(tqdm(train_loader)):
        batch_data, batch_token_starts, batch_labels = batch_samples
        batch_masks = batch_data.gt(0)

        loss = model((batch_data, batch_token_starts),
                     token_type_ids=None, attention_mask=batch_masks, labels=batch_labels)[0]
        train_losses += loss.item()

        # Clear previous gradients, compute gradients of all variables
        model.zero_grad()
        loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config["clip_grad"]) # TODO figure out clip_grad

        # Perform updates using calculated gradients
        optimizer.step()
        scheduler.step()

    train_loss = float(train_losses) / len(train_loader)
    print("Epoch: {}, Train Loss: {}".format(epoch, train_loss))

def train(train_loader, dev_loader, model, optimizer, scheduler, config):

    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, config["training"]["epoch_num"] + 1):
        train_epoch(train_loader, model, optimizer, scheduler, epoch, config)
        val_metrics = evaluate(dev_loader, model, mode='dev')
        val_loss = val_metrics["loss"]
        val_f1 = val_metrics["f1"]

        print("Epoch: {}, dev loss: {}, f1 score: {}".format(epoch, val_metrics['loss'], val_f1))

        improve_f1 = val_f1 - best_val_f1
        if improve_f1 > 1e-5:
            best_val_f1 = val_f1
            model.save_pretrained(config["model"]["model_dir"])
            if improve_f1 < config["training"]["f1_patience"]:
                patience_counter += 1
            else:
                patience_counter = 0
        else:
            patience_counter += 1

        if (patience_counter >= config["training"]["patience_num"] and epoch > config["training"]["min_epoch_num"]):
            print("Best F1 val: {}".format(best_val_f1))
            break

    print("Training finished")


def evaluate(dev_loader, model, config):
    model.eval()
    id2label = config["data"]["id2label"]

    true_tags = []
    pred_tags = []
    sent_data = []
    dev_losses = 0

    with torch.no_grad():
        for idx, batch_samples in enumerate(dev_loader):
            batch_data, batch_token_starts, batch_tags = batch_samples

            batch_masks = batch_data.gt(0) # Get padding mask
            label_masks = batch_tags.gt(-1) # Get padding mask

            # Compute model output and loss
            loss = model((batch_data, batch_token_starts),
                         token_type_ids=None, attention_mask=batch_masks, labels=batch_tags)[0]
            dev_losses += loss.item()

            batch_output = model((batch_data, batch_token_starts),
                                 token_type_ids=None, attention_mask=batch_masks)[0]
            batch_output = model.crf.decode(batch_output, mask=label_masks)

            batch_tags = batch_tags.to('cpu').numpy()
            pred_tags.extend([[id2label.get(idx) for idx in indices] for indices in batch_output])
            true_tags.extend([[id2label.get(idx) for idx in indices if idx > -1] for indices in batch_tags])
    
    assert len(pred_tags) == len(true_tags)

    metrics = {}
    score = f1_score(true_tags, pred_tags, config)
    metrics['f1'] = score['f1']
    metrics['p'] = score['p']
    metrics['r'] = score['r']

    metrics['loss'] = float(dev_losses) / len(dev_loader)
    return metrics

        

