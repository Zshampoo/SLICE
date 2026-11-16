from pathlib import Path
import shutil
from copy import deepcopy

import pandas as pd
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
import random

from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, f1_score, jaccard_score, precision_score, recall_score, roc_auc_score
from src.SIE.Disease_classifier.Data_wrapper_chexpert import *
# from Image_classifier import ChexpertClassifier
from torch.utils.tensorboard import SummaryWriter

from src.SIE.Disease_classifier.biovil_t.get_image_encoder import ChexpertClassifier_
from src.SIE.Disease_classifier.swin_transformer_classifier import build_model_swintransformer

logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
log = logging.getLogger(__name__)

# params
LABEL_SELECTION = 'chexBert'  # 'chexPert' 'chexBert' 'chexpert_plus'
LBAEL_TYPE = '4class' # 'U-zero' 'U-one' '4class'
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:0')
LR = 5e-5 # resnet50 5E-5
SEED = 40
IMAGE_INPUT_SIZE = 512
PERCENTAGE_OF_TRAIN_SET_TO_USE = 1.0
PERCENTAGE_OF_VAL_SET_TO_USE = 1.0 #0.2
BATCH_SIZE = 16 #16
EFFECTIVE_BATCH_SIZE = 64
NUM_WORKERS = 8
EPOCHS = 20
EVALUATE_EVERY_K_STEPS =  1000 # 1000  # how often to evaluate the model on the validation set and log metrics to tensorboard (additionally, model will always be evaluated at end of epoch)
PATIENCE_LR_SCHEDULER = 5  # number of evaluations to wait for val loss to reduce before lr is reduced
THRESHOLD_LR_SCHEDULER = 3e-5
FACTOR_LR_SCHEDULER = 0.5
COOLDOWN_LR_SCHEDULER = 5
RUN = 'phase1'  # train_phase1 4-class / valid.xlsx
path_runs_image_classifier = str(Path(__import__('os').environ.get('DATA_ROOT', './data')))
ANN_PHASE = str(Path(__import__('os').environ.get('DATA_ROOT', './data')))
ANN_VALID = str(Path(__import__('os').environ.get('DATA_ROOT', './data')))
# CheXbert head order: 13 four-way heads, then No Finding binary
CONDITIONS_14 = [
    'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
    'Support Devices', 'No Finding',
]
CLASS4 = {0: 'blank', 1: 'positive', 2: 'negative', 3: 'uncertain'}
from typing import List, Dict
from torch import Tensor
import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def collate_fn(batch: List[Dict[str, Tensor]]):
    # each dict in batch (which is a list) is for a single image and has the keys "image", "labels", "reort"

    # discard images from batch where __getitem__ from custom_image_dataset failed (i.e. returned None)
    # otherwise, whole training loop will stop (even if only 1 image fails to open)
    batch = list(filter(lambda x: x is not None, batch))

    image_shape = batch[0]["image"].size()
    label_shape = batch[0]["label"].size()
    # allocate an empty images_batch tensor that will store all images of the batch
    images_batch = torch.empty(size=(len(batch), *image_shape))
    labels_batch = torch.empty(size=(len(batch), *label_shape), dtype=batch[0]["label"].dtype)
    reports_batch = [""] * len(batch)

    for i, sample in enumerate(batch):
        # remove image tensors from batch and store them in dedicated images_batch tensor
        images_batch[i] = sample.pop("image")
        reports_batch[i] = sample.pop("report")
        labels_batch[i] = sample.pop("label")

    # since batch (which is a list) now only contains dicts with keys "boxes" and "labels", rename it as targets
    # labels = batch

    # create a new batch variable to store images_batch and targets
    batch_new = {}
    batch_new["image"] = images_batch
    batch_new["report"] = reports_batch
    batch_new["label"] = labels_batch

    return batch_new

def get_data_loaders(train_dataset, val_dataset, test_dataset = None):
    def seed_worker(worker_id):
        """To preserve reproducibility for the randomly shuffled train loader."""
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(train_dataset,  collate_fn= collate_fn, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, worker_init_fn=seed_worker, generator=g, pin_memory=True)
    val_loader = DataLoader(val_dataset,  collate_fn= collate_fn, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, collate_fn= collate_fn, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, test_loader

def evaluate_cls(preds, preds_probs, y):

    # calculate jaccard index
    jaccard = jaccard_score(y, preds, average='samples', zero_division=1)

    class_report = classification_report(y, preds, output_dict=True) # dict 18 ['0', '1', …, '13', 'micro avg', 'macro avg', 'weighted avg', 'samples avg']
    # scores = class_report['micro avg']
    scores = class_report['macro avg']
    metrics_per_label = {label: metrics for label, metrics in class_report.items() if label.isdigit()} # dict 14 ['0', '1', …, '13'] ['precision', 'recall', 'f1-score']

    f1 = scores['f1-score']
    rec = scores['recall']
    prec = scores['precision']
    acc = accuracy_score(y.flatten(), preds.flatten())
    try:
        auc = roc_auc_score(y.flatten(), preds_probs.flatten())
    except Exception as e:
        auc = 0.
    return acc, f1, rec, prec, jaccard, auc, metrics_per_label

def fourclass_loss(logits, targets):
    """Sum of per-head cross-entropy. logits is a list of 14 tensors; targets is (B, 14) long."""
    y = targets.long()
    loss = torch.nn.functional.cross_entropy(logits[0], y[:, 0])
    for j, logit in enumerate(logits[1:], start=1):
        loss = loss + torch.nn.functional.cross_entropy(logit, y[:, j])
    return loss


def evaluate_4class(preds, y, class_names):
    """Per-head accuracy and macro F1. preds/y are int arrays (N, 14)."""
    metrics_per_label = {}
    accs, f1s = [], []
    for j, name in enumerate(class_names):
        yt = y[:, j]
        yp = preds[:, j]
        acc_j = accuracy_score(yt, yp)
        f1_j = f1_score(yt, yp, average="macro", zero_division=0)
        prec_j = precision_score(yt, yp, average="macro", zero_division=0)
        rec_j = recall_score(yt, yp, average="macro", zero_division=0)
        accs.append(acc_j)
        f1s.append(f1_j)
        metrics_per_label[str(j)] = {
            "f1-score": f1_j,
            "precision": prec_j,
            "recall": rec_j,
            "accuracy": acc_j,
            "name": name,
        }
    acc = float(np.mean(accs))
    f1 = float(np.mean(f1s))
    return acc, f1, f1, f1, 0.0, 0.0, metrics_per_label


def get_val_loss_and_other_metrics(model, val_dl, writer, overall_steps_taken, criterion):
    model.eval()

    val_loss = 0.0

    num_images = 0

    # tensor for accumulating the number of times a class is detected over all images (will be divided by num_images at the end of get average)
    # sum_class_detected = torch.zeros(29, device=device)

    # tensor for accumulating the intersection area of each class (will be divided by union area of each class at the end of get the IoU for each class)
    # sum_intersection_area_per_class = torch.zeros(29, device=device)

    # tensor for accumulating the union area of each class (will divide the intersection area of each class at the end of get the IoU for each class)
    # sum_union_area_per_class = torch.zeros(29, device=device)
    targets_epoch = []
    preds_epoch = []
    preds_probs_epoch = []
    with torch.no_grad():
        for batch_num, batch in tqdm(enumerate(val_dl), disable=True):
            # "targets" maps to a list of dicts, where each dict has the keys "boxes" and "labels" and corresponds to a single image
            # "boxes" maps to a tensor of shape [29 x 4] and "labels" maps to a tensor of shape [29]
            # note that the "labels" tensor is always sorted, i.e. it is of the form [1, 2, 3, ..., 29] (starting at 1, since 0 is background)
            images, reports, targets = batch.values()

            batch_size = images.size(0)
            num_images += batch_size

            images = images.to(device, non_blocking=True)  # shape (batch_size x 1 x 512 x 512)
            y = targets.to(device, non_blocking=True)  # shape (batch_size x 14)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images) # resnet50
                # logits = model(images).logits  # swintransformer
                if isinstance(logits, list):
                    loss = fourclass_loss(logits, y)
                else:
                    loss = criterion(logits, y) # loss for resnet50 + BCE

            val_loss += loss.item() * batch_size
            # val_loss += loss.item()

            if isinstance(logits, list):
                preds = torch.stack([lg.argmax(dim=1) for lg in logits], dim=1)
                preds_probs = preds.float()
            else:
                ####### for resnet50+BCE #######
                preds_probs = torch.sigmoid(logits)
                preds = preds_probs > 0.5

            preds_probs_epoch.extend(preds_probs.cpu().numpy())
            preds_epoch.extend(preds.cpu().numpy())
            targets_epoch.extend(y.cpu().numpy())

    preds_epoch, preds_probs_epoch, targets_epoch = np.array(preds_epoch), np.array(preds_probs_epoch), np.array(targets_epoch)
    if LBAEL_TYPE == "4class":
        acc, f1, rec, prec, jaccard, auc, metrics_per_label = evaluate_4class(
            preds_epoch, targets_epoch, CONDITIONS_14
        )
    else:
        acc, f1, rec, prec, jaccard, auc, metrics_per_label = evaluate_cls(preds_epoch, preds_probs_epoch, targets_epoch)

    # loss was accumulated as sum(loss * batch_size)
    val_loss /= max(num_images, 1)

    return val_loss,  acc, f1, rec, prec, jaccard, auc, metrics_per_label

def log_stats_to_console(
    train_loss,
    val_loss,
    epoch,
):
    log.info(f"Epoch: {epoch}:")
    log.info(f"\tTrain loss: {train_loss:.3f}")
    log.info(f"\tVal loss: {val_loss:.3f}")

def train_Img_classifier(
    model,
    train_dl,
    val_dl,
    optimizer,
    scaler,
    lr_scheduler,
    epochs,
    criterion,
    weights_folder_path,
    writer,
    class_names,
    label_weights = None,
    start_epoch = 0,
    start_step = 0,
    initial_best_val_loss = None,
    initial_best_state = None,
    initial_best_path = None,
):
    """
    Train a model on train set and evaluate on validation set.
    Saves best model w.r.t. val loss.

    Parameters
    ----------
    model: nn.Module
        The input model to be trained.
    train_dl: torch.utils.data.Dataloder
        The train dataloader to train on.
    val_dl: torch.utils.data.Dataloder
        The val dataloader to validate on.
    optimizer: Optimizer
        The model's optimizer.
    lr_scheduler: torch.optim.lr_scheduler
        The learning rate scheduler to use.
    epochs: int
        Number of epochs to train for.
    patience: int
        Number of epochs to wait for val loss to decrease.
        If patience is exceeded, then training is stopped early.
    weights_folder_path: str
        Path to folder where best weights will be saved.
    writer: torch.utils.tensorboard.SummaryWriter
        Writer for logging values to tensorboard.

    Returns
    -------
    None, but saves model with the overall lowest val loss at the end of every epoch.
    """
    lowest_val_loss = np.inf if initial_best_val_loss is None else float(initial_best_val_loss)

    # the best_model_state is the one where the val loss is the lowest overall
    best_model_state = initial_best_state
    best_model_save_path = initial_best_path
    best_epoch = (start_epoch - 1) if initial_best_state is not None else None

    overall_steps_taken = start_step  # for logging to tensorboard

    # for gradient accumulation
    ACCUMULATION_STEPS = EFFECTIVE_BATCH_SIZE // BATCH_SIZE
    criterion = criterion.to(device)
    for epoch in range(start_epoch, epochs):
        log.info(f"Training epoch {epoch}!")

        train_loss = 0.0
        steps_taken = 0
        Summary = ""
        for num_batch, batch in tqdm(enumerate(train_dl), total= len(train_dl), disable=False):
        # for num_batch, batch in tqdm(enumerate(train_dl), leave=False):
            # break
            # if num_batch > 3:
            #     break
            # batch is a dict with keys "images" "reports" and "targets"
            images, reports, targets = batch.values()

            batch_size = images.size(0)

            images = images.to(device, non_blocking=True)  # shape (batch_size x 3 x 488 x 488)
            targets = targets.to(device, non_blocking=True)  # shape (batch_size x 14)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images) # resnet50
                # logits = model(images).logits # swintransformer
                if isinstance(logits, list):
                    loss = fourclass_loss(logits, targets)
                else:
                    loss = criterion(logits, targets) # loss for resnet 50




            # loss = 0. # loss for swinT
            # for i in range(len(logits)):
            #     # ce_loss = torch.nn.CrossEntropyLoss(weight=label_weights[i]) if label_weights[i] is not None else criterion
            #     loss += criterion(logits[i], targets[:, i].long())

            # Apply sigmoid to get probabilities
            # preds_probs = torch.sigmoid(logits)
            #
            # # Get predictions as boolean values
            # preds = preds_probs > 0.5

            # acc, f1, rec, prec, jaccard, auc, metrics_per_label = evaluate_cls(preds, y)  # train-time eval


            scaler.scale(loss).backward()

            if (num_batch + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * batch_size
            steps_taken += 1
            overall_steps_taken += 1

            # evaluate every k steps and also at the end of an epoch
            if steps_taken >= EVALUATE_EVERY_K_STEPS or (num_batch + 1) == len(train_dl):
                log.info(f"Evaluating at step {overall_steps_taken}!")

                # normalize the train loss by steps_taken
                train_loss /= steps_taken

                val_loss,  acc, f1, rec, prec, jaccard, auc, metrics_per_label = get_val_loss_and_other_metrics(model, val_dl, writer, overall_steps_taken, criterion)

                writer.add_scalars("_loss", {"train_loss": train_loss, "val_loss": val_loss}, overall_steps_taken)
                writer.add_scalars(
                    "overall_metrics",
                    {"val_acc": acc, "val_f1": f1, "val_rec": rec, "val_prec": prec, "val_jaccard": jaccard, "val_auc": auc},
                    overall_steps_taken,
                )

                for class_, metri in zip(class_names, metrics_per_label):
                    writer.add_scalar(f"metric_{class_}", metrics_per_label[metri]["f1-score"], overall_steps_taken)

                current_lr = float(optimizer.param_groups[0]["lr"])
                writer.add_scalar("lr", current_lr, overall_steps_taken)

                log.info(f"Metrics evaluated at step {overall_steps_taken}!")

                # summary
                Summary += f"\nEpoch [{epoch}][{num_batch}/{len(train_dl)}] Step {overall_steps_taken}: Train_Loss {train_loss:.4f}, Valid_Loss {val_loss:.4f}, LR={current_lr:.6f}\n"
                # print
                # tqdm.write(f'Epoch: [{0}][{1}/{2}]\t'
                #       'Step: {3}\t'
                #       'Train Loss:{4}\t'
                #       'Valid Loss:{5}\t'.format(epoch, num_batch, len(train_dl), overall_steps_taken, train_loss, val_loss))
                # tqdm.write('**********Detections per class**********')
                # summary
                Summary += f"\n**********Classification Result on Validation Dataset**********\n"
                Summary += f"Val_acc:{acc:.6f}, Val_f1:{f1:.6f}, Val_rec:{rec:.6f}, Val_prec:{prec:.6f}, Val_jaccard:{jaccard:.6f}, Val_auc:{auc:.6f}\n"
                for class_, metri in zip(class_names, metrics_per_label):
                    f1_class, precision_class, recall_class = metrics_per_label[metri]['f1-score'], metrics_per_label[metri]['precision'], metrics_per_label[metri]['recall']
                    Summary += f" | Class_{class_} f1:{f1_class:.6f}, precision:{precision_class:.6f}, recall:{recall_class:.6f} |\n"
                model.train()

                # decrease lr if val loss has not decreased after certain number of evaluations
                lr_scheduler.step(val_loss)

                if val_loss < lowest_val_loss:
                    lowest_val_loss = val_loss
                    best_epoch = epoch
                    best_model_save_path = os.path.join(
                        weights_folder_path, f"val_loss_{lowest_val_loss:.3f}_epoch_{epoch}_step_{overall_steps_taken}_acc_{acc:.6f}_f1_{f1:.6f}_auc_{auc:.6f}.pth"
                    )
                    best_model_state = deepcopy(model.state_dict())

                # log to console at the end of an epoch
                if (num_batch + 1) == len(train_dl):
                    log_stats_to_console(train_loss, val_loss, epoch)

                # reset values
                train_loss = 0.0
                steps_taken = 0

        # print
        tqdm.write(Summary)
        run_folder_path = os.path.join(path_runs_image_classifier, f"run_{RUN}_{LABEL_SELECTION}_{LBAEL_TYPE}")
        with open(run_folder_path + "/training_summary_epoch{}.txt".format(epoch), "w") as f:
            f.write(Summary)
        # save the current best model weights at the end of each epoch
        torch.save(best_model_state, best_model_save_path)

    log.info("Finished training!")
    log.info(f"Lowest overall val loss: {lowest_val_loss:.3f} at epoch {best_epoch}")
    return None

def create_run_folder(resume=False):
    """
    Run folder will contain a folder for saving the trained weights, a folder for the tensorboard files
    as well as a config file that specifies the overall parameters used for training.
    """
    run_folder_path = os.path.join(path_runs_image_classifier, f"run_{RUN}_{LABEL_SELECTION}_{LBAEL_TYPE}")
    weights_folder_path = os.path.join(run_folder_path, "weights")
    tensorboard_folder_path = os.path.join(run_folder_path, "tensorboard")

    if os.path.exists(run_folder_path):
        if resume:
            log.info(f"Resume into existing run folder {run_folder_path}.")
            return weights_folder_path, tensorboard_folder_path, os.path.join(run_folder_path, "run_config.txt")
        log.error(f"Folder to save run {RUN} already exists at {run_folder_path}.")
        shutil.rmtree(run_folder_path)
        log.error("Delete the folder.")
        # log.error("Delete the folder or change the run number.")
        # return None

    os.mkdir(run_folder_path)
    os.mkdir(weights_folder_path)
    os.mkdir(tensorboard_folder_path)

    log.info(f"Run {RUN} folder created at {run_folder_path}.")

    config_file_path = os.path.join(run_folder_path, "run_config.txt")
    config_parameters = {
        "SEED": SEED,
        "LABEL_USE": LABEL_SELECTION,
        "LABEL_TYPE": LBAEL_TYPE,
        "IMAGE_INPUT_SIZE": IMAGE_INPUT_SIZE,
        "PERCENTAGE_OF_TRAIN_SET_TO_USE": PERCENTAGE_OF_TRAIN_SET_TO_USE,
        "PERCENTAGE_OF_VAL_SET_TO_USE": PERCENTAGE_OF_VAL_SET_TO_USE,
        "BATCH_SIZE": BATCH_SIZE,
        "EFFECTIVE_BATCH_SIZE": EFFECTIVE_BATCH_SIZE,
        "NUM_WORKERS": NUM_WORKERS,
        "EPOCHS": EPOCHS,
        "LR": LR,
        "EVALUATE_EVERY_K_STEPS": EVALUATE_EVERY_K_STEPS,
        "PATIENCE_LR_SCHEDULER": PATIENCE_LR_SCHEDULER,
        "FACTOR_LR_SCHEDULER": FACTOR_LR_SCHEDULER,
        "THRESHOLD_LR_SCHEDULER": THRESHOLD_LR_SCHEDULER,
        "COOLDOWN_LR_SCHEDULER": COOLDOWN_LR_SCHEDULER
    }

    with open(config_file_path, "w") as f:
        f.write(f"RUN {RUN}:\n")
        for param_name, param_value in config_parameters.items():
            f.write(f"\t{param_name}: {param_value}\n")

    return weights_folder_path, tensorboard_folder_path, config_file_path

def label_statistics(df, chexlabels, type=None):
    label_dict = {}
    abnormal_percent = {}
    for label in chexlabels:
        label_dict[label] = {}
        abnormal_percent[label] = 0
    for index, row in df.iterrows():
        for label in chexlabels:
            result = row[label]
            if not np.isnan(result):
                if result not in label_dict[label].keys():
                    label_dict[label][result] = 1
                else:
                    label_dict[label][result] += 1
                if result != 0.0:  # neither nan nor 0.0
                    abnormal_percent[label] += 1

    print(f"{type} dataset label distribution (by {LABEL_SELECTION}) (all count {df.shape[0]})")
    for label_key in label_dict.keys():
        for result in label_dict[label_key].keys():
            print(f"{label_key}|{result}: {label_dict[label_key][result]}", end=' ')
        print(f"Abnormal: {abnormal_percent[label_key]}")
        print('\n')

    return label_dict


if __name__ == '__main__':
    # 26.89M
    # model = ChexpertClassifier_(num_classes=14)
    # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Number of params: {n_parameters}")
    # print(f"Number of params: {n_parameters:,} ({n_parameters / 1e6:.2f}M)")

    # phase1 train + valid.xlsx. Heads follow CheXbert order, No Finding last (binary).
    ycols = [f"{d}_y" for d in CONDITIONS_14]
    usecols_train = ["dicom_id", "SPLIT_0414", "mimic_image_file_path"] + ycols
    log.info(f"reading train {ANN_PHASE}")
    df_all = pd.read_excel(ANN_PHASE, usecols=usecols_train, engine="openpyxl")
    df_train = df_all[df_all["SPLIT_0414"] == "train_phase1"].reset_index(drop=True)
    log.info(f"reading valid {ANN_VALID}")
    df_valid = pd.read_excel(
        ANN_VALID,
        usecols=["dicom_id", "mimic_image_file_path"] + ycols,
        engine="openpyxl",
    )
    chexlabels = ycols
    log.info(f"phase1 train={len(df_train)} valid={len(df_valid)}")
    assert len(df_train) == 111936, f"train_phase1 expected 111936, got {len(df_train)}"
    assert len(df_valid) == 5958, f"valid expected 5958, got {len(df_valid)}"

    def _count_polarity(frame, cols, title):
        log.info(f"{title} n={len(frame)}")
        print(f"\n{title} n={len(frame)}")
        print(f"{'disease':28s} {'blank':>8s} {'positive':>8s} {'negative':>8s} {'uncertain':>9s}")
        for col in cols:
            s = frame[col]
            n_blank = int(s.isna().sum())
            n_pos = int((s == 1).sum())
            n_neg = int((s == 0).sum())
            n_unc = int((s == -1).sum())
            name = col[:-2] if col.endswith("_y") else col
            print(f"{name:28s} {n_blank:8d} {n_pos:8d} {n_neg:8d} {n_unc:9d}")

    _count_polarity(df_train, chexlabels, "train_phase1")
    _count_polarity(df_valid, chexlabels, "valid")

    train_dataset = Chexpert_Dataset(
        df_train, chexlabels, split="train", root_dir=None, shape=488,
        label_type=LBAEL_TYPE, loss_weighting="none", use_augs=True,
    )
    valid_dataset = Chexpert_Dataset(
        df_valid, chexlabels, split="val", root_dir=None, shape=488,
        label_type=LBAEL_TYPE, loss_weighting="none", use_augs=False,
    )

    class_weights = None
    train_dataloader, val_dataloader, _ = get_data_loaders(train_dataset, valid_dataset, test_dataset=None)

    num_classes = len(chexlabels)
    class_names = CONDITIONS_14

    # 14 independent CE heads; No Finding is 2-way, the other 13 are 4-way
    criterion = CrossEntropyLoss()
    model = ChexpertClassifier_(num_classes=num_classes, head_mode="chexbert4")

    resume_ckpt = os.environ.get("RESUME_CKPT", "").strip()
    resume_epoch = int(os.environ.get("RESUME_EPOCH", "0"))
    resume_step = int(os.environ.get("RESUME_STEP", "0"))
    resume_best_raw = os.environ.get("RESUME_BEST_VAL_LOSS", "").strip()
    resume_best_val = float(resume_best_raw) if resume_best_raw else None
    resumed_state = None
    if resume_ckpt:
        resumed_state = torch.load(resume_ckpt, map_location="cpu")
        model.load_state_dict(resumed_state)
        log.info(
            f"Loaded resume checkpoint {resume_ckpt} "
            f"(next epoch {resume_epoch}, step {resume_step}, best val loss {resume_best_val})"
        )

    model.to(device, non_blocking=True)
    model.train()

    weights_folder_path, tensorboard_folder_path, config_file_path = create_run_folder(resume=bool(resume_ckpt))
    writer = SummaryWriter(log_dir=tensorboard_folder_path)
    opt = AdamW(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler()
    lr_scheduler = ReduceLROnPlateau(opt, mode="min", factor=FACTOR_LR_SCHEDULER, patience=PATIENCE_LR_SCHEDULER, threshold= THRESHOLD_LR_SCHEDULER, cooldown=COOLDOWN_LR_SCHEDULER)
    # writer = SummaryWriter(log_dir=tensorboard_folder_path)
    print("\nStarting training!\n")

    train_Img_classifier(model = model,
                         train_dl=train_dataloader,
                         val_dl=val_dataloader,
                         optimizer=opt,
                         scaler=scaler,
                         lr_scheduler=lr_scheduler,
                         criterion = criterion,
                         epochs=EPOCHS,
                         weights_folder_path=weights_folder_path,
                         writer=writer,
                         class_names = class_names,
                         label_weights = class_weights,
                         start_epoch=resume_epoch if resume_ckpt else 0,
                         start_step=resume_step if resume_ckpt else 0,
                         initial_best_val_loss=resume_best_val,
                         initial_best_state=resumed_state,
                         initial_best_path=resume_ckpt if resume_ckpt else None)