import torch
from training.pseudo_labeling import predict_unlabeled
from training.train_configs import model_configs
from preprocessing.dataloader import load_data, load_unlabeled_datat
from models.BertCrf import BertCrf

# Define languages
languages = {
    "fo": load_data("data/labeled/fo_data.csv"),
    "co": load_data("data/labeled/co_data.csv"),
    "hsb": load_data("data/labeled/hsb_data.csv"),
    "bh": load_data("data/labeled/bh_data.csv"),
    "cv": load_data("data/labeled/cv_data.csv"),
    "mg": load_data("data/labeled/mg_data.csv"),
}

# Unlabeled data
unlabeled_texts = {
    "fo": load_unlabeled_datat("data/un_labeled/fo_texts.txt"),
    "co": load_unlabeled_datat("data/labeled/co_texts.txt"),
    "hsb": load_unlabeled_datat("data/labeled/hsb_texts.txt"),
    "bh": load_unlabeled_datat("data/labeled/bh_texts.txt"),
    "cv": load_unlabeled_datat("data/labeled/cv_texts.txt"),
    "mg": load_unlabeled_datat("data/labeled/mg_texts.txt"),
}

baseline_model_performance = {}

# Define teacher model
teacher_model_name = "BERT-CRF"
teacher_model = BertCrf(num_tags=model_configs[teacher_model_name].NUM_TAGS)

teacher_model.load_state_dict(
    torch.load("fo_baseline_pretrained.pth", weights_only=True)
)

# Define student model
student_model_name = "BERT-CRF"
student_model = BertCrf(num_tags=model_configs[student_model_name].NUM_TAGS)

# Just testing
pred_tags = predict_unlabeled(
    teacher_model, student_model, unlabeled_texts["fo"], 32, torch.device("cuda")
)
