import torch.optim as optim


def setup_optimizer(model, config):
    """
    Setups up optimizer for different models with different learning rates for each part
    """
    param_groups = []
    if hasattr(model, "bert") or hasattr(model, "roberta"):
        param_groups.append(
            {
                "params": model.bert.parameters()
                if hasattr(model, "bert")
                else model.roberta.parameters(),
                "lr": config["training"]["bert_learning_rate"],
            }
        )

    if hasattr(model, "lstm"):
        param_groups.append(
            {
                "params": model.lstm.parameters(),
                "lr": config["training"]["lstm_learning_rate"],
            }
        )

    if hasattr(model, "crf"):
        param_groups.append(
            {
                "params": model.crf.parameters(),
                "lr": config["training"]["crf_learning_rate"],
            }
        )

    optimizer = optim.Adam(param_groups)
    return optimizer
