import torch
from training.pseudo_labeling import pseudo_labeling_train
from training.train_configs import model_configs
from preprocessing.dataloader import load_data, load_unlabeled_data
from models.BertBilstmCrf import BertBilstmCrf

# Define languages
languages = {
    "mg": load_data("data/labeled/mg_data.csv"),
    "fo": load_data("data/labeled/fo_data.csv"),
    "co": load_data("data/labeled/co_data.csv"),
    "hsb": load_data("data/labeled/hsb_data.csv"),
    "bh": load_data("data/labeled/bh_data.csv"),
    "cv": load_data("data/labeled/cv_data.csv"),
}

# Unlabeled data
unlabeled_texts = {
    "mg": load_unlabeled_data("data/unlabeled/mg_texts.txt"),
    "fo": load_unlabeled_data("data/unlabeled/fo_texts.txt"),
    "co": load_unlabeled_data("data/unlabeled/co_texts.txt"),
    "hsb": load_unlabeled_data("data/unlabeled/hsb_texts.txt"),
    "bh": load_unlabeled_data("data/unlabeled/bh_texts.txt"),
    "cv": load_unlabeled_data("data/unlabeled/cv_texts.txt"),
}


# Define teacher model
teacher_model_name = "BERT-Bilstm-CRF"
teacher_model = BertBilstmCrf(num_tags=model_configs[teacher_model_name].NUM_TAGS)

teacher_model.load_state_dict(
    torch.load("src/models/pretrained/fo_baseline_pretrained.pth", weights_only=True)
)

# Define student model
student_model_name = "BERT-Bilstm-CRF"
student_model_type = BertBilstmCrf

# Run with pseudo-labeling
train_f1, test_f1 = pseudo_labeling_train(
    teacher_model,
    student_model_type,
    unlabeled_texts["fo"].head(300),
    languages["fo"],
    save_model=f"src/models/pretrained/{'fo'}_pseudo_pretrained.pth",
    config=model_configs[student_model_name],
)
