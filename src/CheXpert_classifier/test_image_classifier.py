from pathlib import Path
import ast
import os.path
import re
import shutil
from copy import deepcopy

import numpy as np
import pandas as pd
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
import random

from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, jaccard_score, roc_auc_score
from Data_wrapper_chexpert import *
from Image_classifier import ChexpertClassifier
from torch.utils.tensorboard import SummaryWriter

from src.CheXpert_classifier.Data_wrapper_chexpert import Chexpert_Dataset_test
from src.CheXpert_classifier.biovil_t.get_image_encoder import ChexpertClassifier_
from src.CheXpert_classifier.swin_transformer_classifier import build_model_swintransformer

logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
log = logging.getLogger(__name__)

# params

# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:0')
LR = 5e-5
SEED = 40
IMAGE_INPUT_SIZE = 512
PERCENTAGE_OF_TRAIN_SET_TO_USE = 1.0
PERCENTAGE_OF_VAL_SET_TO_USE = 1.0 #0.2
BATCH_SIZE = 32 #16
EFFECTIVE_BATCH_SIZE = 64
NUM_WORKERS = 8
EPOCHS = 20
EVALUATE_EVERY_K_STEPS = 1000  # how often to evaluate the model on the validation set and log metrics to tensorboard (additionally, model will always be evaluated at end of epoch)
PATIENCE_LR_SCHEDULER = 5  # number of evaluations to wait for val loss to reduce before lr is reduced
THRESHOLD_LR_SCHEDULER = 5e-5
FACTOR_LR_SCHEDULER = 0.5
COOLDOWN_LR_SCHEDULER = 5
RUN = 418
from src.path_datasets_and_weights import path_runs_image_classifier
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
    labels_batch = torch.empty(size=(len(batch), *label_shape))
    reports_batch = [""] * len(batch)
    fileid_batch = [""] * len(batch)

    for i, sample in enumerate(batch):
        # remove image tensors from batch and store them in dedicated images_batch tensor
        images_batch[i] = sample.pop("image")
        reports_batch[i] = sample.pop("report")
        labels_batch[i] = sample.pop("label")
        fileid_batch[i] = sample.pop("file_id")

    # since batch (which is a list) now only contains dicts with keys "boxes" and "labels", rename it as targets
    # labels = batch

    # create a new batch variable to store images_batch and targets
    batch_new = {}
    batch_new["image"] = images_batch
    batch_new["report"] = reports_batch
    batch_new["label"] = labels_batch
    batch_new["file_id"] = fileid_batch

    return batch_new

def get_test_loader(test_dataset):
    def seed_worker(worker_id):
        """To preserve reproducibility for the randomly shuffled train loader."""
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(SEED)

    test_loader = DataLoader(test_dataset, collate_fn= collate_fn, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    return test_loader

def evaluate_cls(preds, preds_probs, y, selected_indices = None):

    # calculate jaccard index
    jaccard = jaccard_score(y, preds, average='samples', zero_division=1)

    class_report = classification_report(y, preds, output_dict=True) # dict 18 ['0', '1', …, '13', 'micro avg', 'macro avg', 'weighted avg', 'samples avg']
    # scores = class_report['micro avg']
    scores = class_report['macro avg']
    metrics_per_label = {label: metrics for label, metrics in class_report.items() if label.isdigit()} # dict 14 ['0', '1', …, '13'] ['precision', 'recall', 'f1-score']

    f1 = scores['f1-score']
    rec = scores['recall']
    prec = scores['precision']

    acc_all = accuracy_score(y.flatten(), preds.flatten())

    # 对每一行检查是否所有标签都预测对了
    correct_per_sample = np.all(y == preds, axis=1)  # shape: (n_samples,)

    # 然后求平均
    sample_accuracy = correct_per_sample.mean()

    # try:
    auc_all = roc_auc_score(y.flatten(), preds_probs.flatten())

    acc_per_class = []
    for i in range(y.shape[1]):
        acc_ = accuracy_score(y[:, i], preds[:, i])
        acc_per_class.append(acc_)

    auc_per_class = []
    for i in range(y.shape[1]):
        auc_ = roc_auc_score(y[:, i], preds_probs[:, i])
        auc_per_class.append(auc_)


    # except Exception as e:
    #     auc = 0.
    return acc_all, f1, rec, prec, jaccard, auc_all, metrics_per_label, acc_per_class, auc_per_class, sample_accuracy

def log_stats_to_console(
    train_loss,
    val_loss,
    epoch,
):
    log.info(f"Epoch: {epoch}:")
    log.info(f"\tTrain loss: {train_loss:.3f}")
    log.info(f"\tVal loss: {val_loss:.3f}")

def test_Img_classifier(
    model,
    test_dl,
    label_names,
    criterion,
    data_origin,
    write_path,
    label_selection,
    label_type,
    write_type = 'test'
):

    overall_steps_taken = 0  # for logging to tensorboard

    # for gradient accumulation
    ACCUMULATION_STEPS = EFFECTIVE_BATCH_SIZE // BATCH_SIZE
    criterion = criterion.to(device)

    test_loss = 0.0

    num_images = 0
    targets_epoch = []
    preds_epoch = []
    preds_probs_epoch = []

    # Summary = ""
    result_tuples = []
    with torch.no_grad():
        model.eval()
        for batch_num, batch in tqdm(enumerate(test_dl), leave=False, total=len(test_dl)):
            # if batch_num >10 :
            #     break
            # "targets" maps to a list of dicts, where each dict has the keys "boxes" and "labels" and corresponds to a single image
            # "boxes" maps to a tensor of shape [29 x 4] and "labels" maps to a tensor of shape [29]
            # note that the "labels" tensor is always sorted, i.e. it is of the form [1, 2, 3, ..., 29] (starting at 1, since 0 is background)
            images, reports, targets, file_ids = batch.values()

            batch_size = images.size(0)
            num_images += batch_size

            images = images.to(device, non_blocking=True)  # shape (batch_size x 1 x 512 x 512)
            y = targets.to(device, non_blocking=True)  # shape (batch_size x 14)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)# resnet50
                # logits = model(images).logits # swintransformer

                loss = criterion(logits, y)

            # Apply sigmoid to get probabilities
            preds_probs = torch.sigmoid(logits) # (64,14)

            # Get predictions as boolean values
            preds = preds_probs > 0.5

            for i in range(batch_size):
                pred_is_true = np.all(y[i].detach().cpu().numpy() == preds[i].detach().cpu().numpy())
                result_tuples.append([file_ids[i],
                                      preds_probs[i].detach().cpu().numpy(),
                                      preds[i].detach().cpu().numpy(),
                                      pred_is_true])

            test_loss += loss.item() * batch_size



            preds_probs_epoch.extend(preds_probs.cpu().numpy())
            preds_epoch.extend(preds.cpu().numpy())
            targets_epoch.extend(y.cpu().numpy())

    preds_epoch, preds_probs_epoch, targets_epoch = (np.array(preds_epoch), np.array(preds_probs_epoch), np.array(targets_epoch))
    acc, f1, rec, prec, jaccard, auc, metrics_per_label, acc_per_class, auc_per_class, sample_accuracy = evaluate_cls(preds_epoch, preds_probs_epoch, targets_epoch)

    test_loss /= len(test_dl)

    # summary
    # Summary += f"\nEpoch [{epoch}][{num_batch}/{len(train_dl)}] Step {overall_steps_taken}: Train_Loss {train_loss:.4f}, Valid_Loss {val_loss:.4f}, LR={current_lr:.6f}\n"
    # print
    # tqdm.write(f'Epoch: [{0}][{1}/{2}]\t'
    #       'Step: {3}\t'
    #       'Train Loss:{4}\t'
    #       'Valid Loss:{5}\t'.format(epoch, num_batch, len(train_dl), overall_steps_taken, train_loss, val_loss))
    # tqdm.write('**********Detections per class**********')
    # summary
    # Summary += f"\n**********Classification Result on Validation Dataset**********\n"
    # Summary += f"Val_acc:{acc:.6f}, Val_f1:{f1:.6f}, Val_rec:{rec:.6f}, Val_prec:{prec:.6f}, Val_jaccard:{jaccard:.6f}, Val_auc:{auc:.6f}\n"
    # for class_, metri in zip(class_names, metrics_per_label):
    #     f1_class, precision_class, recall_class = metrics_per_label[metri]['f1-score'], metrics_per_label[metri]['precision'], metrics_per_label[metri]['recall']
    #     Summary += f" | Class_{class_} f1:{f1_class:.6f}, precision:{f1_class:.6f}, recall:{f1_class:.6f} |\n"
    # model.train()

    save_overall_scores(label_names, num_images, metrics_per_label, acc, f1, rec, prec, jaccard, auc, acc_per_class, auc_per_class, sample_accuracy, test_loss, write_type, write_path, label_selection, label_type)
    save_persample_result(data_origin, label_names, result_tuples, write_path, label_selection, label_type, write_type)

    log.info("Finished testing!")
    return None

def save_overall_scores(label_names, num_images, metrics_per_label, acc, f1, rec, prec, jaccard, auc, acc_per_class, auc_per_class, sample_accuracy, test_loss, write_type, write_path, label_selection, label_type):
    cls_file_path = write_path + '/{}_cls_result_{}_{}_mimic15w.txt'.format(write_type, label_selection, label_type)
    with open(cls_file_path, 'w') as f:
        f.write(f'{write_type} sample count: {num_images}\n')
        f.write(f'Test loss: {test_loss:.6f}\n')
        f.write('*************** Overall Classification Scores ***************\n')
        f.write(f'Sample-level ACC: {sample_accuracy:.6f}\n')
        f.write(f'Label-level ACC: {acc:.6f}\n')
        f.write(f'AUC: {auc:.6f}\n')
        f.write(f'F1-score: {f1:.6f}\n')
        f.write(f'Recall: {rec:.6f}\n')
        f.write(f'Precision: {prec:.6f}\n')
        f.write(f'Jaccard: {jaccard:.6f}\n')
        f.write('*************** Class-wise Classification Scores ***************\n')
        for index, (class_, metri) in enumerate(zip(label_names, metrics_per_label)):
            support_num = metrics_per_label[metri]['support']
            f1_class, precision_class, recall_class = metrics_per_label[metri]['f1-score'], metrics_per_label[metri]['precision'], metrics_per_label[metri]['recall']
            acc_class = acc_per_class[index]
            auc_class = auc_per_class[index]
            f.write(f'{class_}-{support_num} \n acc:{acc_class:.6f} \t auc:{auc_class:.6f} \t f1:{f1_class:.6f} \t precision:{precision_class:.6f} \t recall:{recall_class:.6f}\n')
    return cls_file_path

def save_persample_result(df, label_names, result, writepath_root, label_selection, label_type, writetype='test'):
    # assert len(df) == len(result), "shape not same!"
    # df_result = pd.DataFrame(result, columns=['mimic_image_file_path', 'predicted_probs', 'predicted_class', 'is_true']) #["mimic_image_file_path", "bbox_coordinates", "bbox_labels", "bbox_is_abnormal", "all_reference_report"]
    # df_result = pd.DataFrame(result, columns=['Path', 'predicted_probs', 'predicted_class', 'is_true'])  # iuxray
    df_result = pd.DataFrame(result, columns=['image_path', 'predicted_probs', 'predicted_class', 'is_true'])
    # logit
    # logits_array = np.stack(df_result['predicted_logits'].values)
    # logits_tensor = torch.tensor(logits_array, dtype=torch.float32)
    # sigmoid = nn.Sigmoid()
    # probs = list(sigmoid(logits_tensor).numpy()) # 对应于正类（label=1， abnormal）的置信度分数
    # df_result['predicted_abnormal_regions_probs'] = probs

    # combination
    result2ab = {True:'Abnormal', False:'Normal'}
    pred_info_tuple = []
    for index, row in df_result.iterrows():
        sample_tuple = []
        for j in range(len(label_names)):
            label_name = label_names[j].split('_')[0] #
            pred = row['predicted_class'][j]
            prob = row['predicted_probs'][j]
            pred_tf = result2ab[pred]
            if pred_tf == 'Abnormal':
                sample_tuple.append([label_name, pred_tf, prob])
            elif pred_tf == 'Normal':
                sample_tuple.append([label_name, pred_tf, 1-prob])
            else:
                print('error value of pred')
        pred_info_tuple.append(sample_tuple)
    df_result['pred_info'] = pred_info_tuple

    # df_new = pd.merge(df, df_result, on='mimic_image_file_path', how='outer') # mimic
    # df_new = pd.merge(df, df_result, on='Path', how='outer')  # iuxray
    df_new = pd.merge(df, df_result, on='image_path', how='outer')  # chexpertplus
    df_new = df_new.astype(str)

    path = writepath_root + '/persample_{}_result_{}_{}_mimicresnet15w.xlsx'.format(writetype, label_selection, label_type)
    df_cleaned = df_new.applymap(lambda x: x.replace('\n', ' ') if isinstance(x, str) else x)
    df_cleaned.to_excel(path, index=False)
    return path

def test_classifier_chexpert_5_200(
    model,
    test_dl,
    label_names,
    criterion,
    write_path,
    label_selection,
    label_type,
    write_type,
    selected_indices
):

    overall_steps_taken = 0  # for logging to tensorboard

    criterion = criterion.to(device)

    test_loss = 0.0

    num_images = 0
    targets_epoch = []
    preds_epoch = []
    preds_probs_epoch = []

    # Summary = ""
    result_tuples = []
    with torch.no_grad():
        model.eval()
        for batch_num, sample in tqdm(enumerate(test_dl), leave=False, total=len(test_dl)):
            # if batch_num >10 :
            #     break
            # "targets" maps to a list of dicts, where each dict has the keys "boxes" and "labels" and corresponds to a single image
            # "boxes" maps to a tensor of shape [29 x 4] and "labels" maps to a tensor of shape [29]
            # note that the "labels" tensor is always sorted, i.e. it is of the form [1, 2, 3, ..., 29] (starting at 1, since 0 is background)
            images, targets, file_ids = sample.values()

            batch_size = images.size(0)
            num_images += batch_size
            # images = images.to(torch.float32)
            images = images.to(device, non_blocking=True)  # shape (batch_size x 1 x 512 x 512)
            y = targets.to(device, non_blocking=True)  # shape (batch_size x 14)

            # with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(images)

            loss = criterion(logits, y)

            # selected
            logits = logits[:, selected_indices]
            y = y[:, selected_indices]

            # Apply sigmoid to get probabilities
            preds_probs = torch.sigmoid(logits) # (64,14)

            # Get predictions as boolean values
            preds = preds_probs > 0.5

            for i in range(batch_size):
                pred_is_true = np.all(y[i].detach().cpu().numpy() == preds[i].detach().cpu().numpy())
                result_tuples.append([file_ids[i],
                                      preds_probs[i].detach().cpu().numpy(),
                                      preds[i].detach().cpu().numpy(),
                                      pred_is_true])

            test_loss += loss.item() * batch_size



            preds_probs_epoch.extend(preds_probs.cpu().numpy())
            preds_epoch.extend(preds.cpu().numpy())
            targets_epoch.extend(y.cpu().numpy())

    preds_epoch, preds_probs_epoch, targets_epoch = (np.array(preds_epoch), np.array(preds_probs_epoch), np.array(targets_epoch))
    acc, f1, rec, prec, jaccard, auc, metrics_per_label, acc_per_class, auc_per_class, sample_accuracy = evaluate_cls(preds_epoch, preds_probs_epoch, targets_epoch, selected_indices)

    test_loss /= len(test_dl)
    selected_label_names = [label_names[i] for i in selected_indices]

    save_overall_scores(selected_label_names, num_images, metrics_per_label, acc, f1, rec, prec, jaccard, auc, acc_per_class, auc_per_class, sample_accuracy, test_loss, write_type, write_path, label_selection, label_type)
    save_persample_result(data_origin, label_names, result_tuples, write_path, label_selection, label_type, write_type)

    log.info("Finished testing!")
    return None

def create_run_folder():
    """
    Run folder will contain a folder for saving the trained weights, a folder for the tensorboard files
    as well as a config file that specifies the overall parameters used for training.
    """
    run_folder_path = os.path.join(path_runs_image_classifier, f"run_{RUN}")
    weights_folder_path = os.path.join(run_folder_path, "weights")
    tensorboard_folder_path = os.path.join(run_folder_path, "tensorboard")

    if os.path.exists(run_folder_path):
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
    for label in chexlabels:
        label_dict[label] = {}
    for index, row in df.iterrows():
        for label in chexlabels:
            result = row[label]
            if not np.isnan(result):
                if result not in label_dict[label].keys():
                    label_dict[label][result] = 1
                else:
                    label_dict[label][result] += 1
            else:
                if 'nan' not in label_dict[label].keys():
                    label_dict[label]['nan'] = 1
                else:
                    label_dict[label]['nan'] += 1

    print(f"{type} dataset label distribution (by {LABEL_SELECTION})")
    for label_key in label_dict.keys():
        for result in label_dict[label_key].keys():
            print(f"{label_key}|{result}: {label_dict[label_key][result]}", end=' ')
        print('\n')

    return label_dict

def fix_and_eval(s):
    s_fixed = re.sub(r'(?<=[a-zA-Z\]])\s+(?=[a-zA-Z\[])', ', ', s.strip())
    return ast.literal_eval(s_fixed)

def data_merge():
    df = pd.read_excel(os.environ.get('TEST_XLSX', str(Path(os.environ.get('DATA_ROOT','./data'))/'evals'/'test.xlsx')), engine='openpyxl')
    pred_result = np.stack(df['predicted_abnormal_regions'].apply(fix_and_eval).values)
    target= np.stack(df['bbox_is_abnormal'].apply(fix_and_eval).values)
    is_true = np.all(pred_result == target, axis=1)
    acc = np.mean(is_true)
    df.insert(0, 'is_true_regionabnormal', is_true)
    # df = pd.read_excel('./data/<path>', engine='openpyxl')
    # df2 = pd.read_excel('./data/<path>', engine = 'openpyxl')
    # df3 = pd.read_excel('./data/<path>', engine='openpyxl')
    df3 = pd.read_excel(os.environ.get('TEST_XLSX_3', str(Path(os.environ.get('DATA_ROOT','./data'))/'evals'/'test3.xlsx')), engine='openpyxl')
    df_new = pd.merge(df, df3, on='mimic_image_file_path', how='left')

    df_new.to_excel(os.environ.get('TEST_OUT_XLSX', str(Path(os.environ.get('DATA_ROOT','./data'))/'evals'/'out.xlsx')), index=False)

    return None

def test_chexpert_5_200():

    # path
    LABEL_SELECTION = 'U-one' # 'U-zero'
    LABEL_TYPE = 'chexbert' # chexpert chexbert
    WRITETYPE = 'test_chexpert_5_200'
    WRITEPATH = os.environ.get('WRITEPATH', str(Path(os.environ.get('DATA_ROOT','./data'))/'evals'))
    IMAGE_ROOT_PATH = str(Path(__import__('os').environ.get('DATA_ROOT', './data')))
    if not os.path.exists(WRITEPATH):
        os.makedirs(WRITEPATH)
    else:
        print(WRITEPATH, 'already exists')
    # dataset
    USECOLS = [ "Path",
                "No Finding",
                "Enlarged Cardiomediastinum",
                "Cardiomegaly",
                "Lung Opacity",
                "Lung Lesion",
                "Edema",
                "Consolidation",
                "Pneumonia",
                "Atelectasis",
                "Pneumothorax",
                "Pleural Effusion",
                "Pleural Other",
                "Fracture",
                "Support Devices"]
    df = pd.read_csv(str(Path(__import__('os').environ.get('DATA_ROOT', './data'))), usecols=USECOLS, encoding='utf-8')
    df['Path'] = df['Path'].apply(lambda x: os.path.join(IMAGE_ROOT_PATH, x))
    chexlabels = USECOLS[1:]
    test_dataset = Chexpert_5_200_testset(df, chexlabels, label_type=LABEL_SELECTION)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    num_classes = len(chexlabels)
    # loss
    criterion = BCEWithLogitsLoss()

    # model
    model = ChexpertClassifier(num_classes=num_classes)
    # path_to_best_image_classifier_weights = './data/<path>'  # chexbert uzero
    # path_to_best_image_classifier_weights = './data/<path>' # chexbert + uone 15w train best model

    # path_to_best_image_classifier_weights = './data/<path>'
    path_to_best_image_classifier_weights = str(Path(__import__('os').environ.get('DATA_ROOT', './data')))
    load_model = torch.load(path_to_best_image_classifier_weights)
    model.load_state_dict(load_model)
    model.to(device, non_blocking=True)
    print('Loaded weights from {}'.format(path_to_best_image_classifier_weights))

    test_classifier_chexpert_5_200(model=model,
                        test_dl=test_loader,
                        label_names=chexlabels,
                        criterion=criterion,
                        write_path=WRITEPATH,
                        write_type=WRITETYPE,
                        label_selection=LABEL_SELECTION,
                        label_type=LABEL_TYPE,
                        selected_indices = [2, 5, 6, 8, 10])

def analysis_iuxray():
    with open(str(Path(__import__('os').environ.get('DATA_ROOT', './data'))), 'r') as f:
        json_iux = json.load(f)
    df_test = pd.read_excel(str(Path(__import__('os').environ.get('DATA_ROOT', './data'))))
    ids = df_test['Path'].tolist()
    missing_ids = []
    count = 0
    for id in ids:
        flag = False
        for item in json_iux:
            if id == item['id']:
                print(item['view'])
                flag = True
                break
                count += 1
                flag = True
                break
        if not flag:
            missing_ids.append(id)
    return count, missing_ids


if __name__ == '__main__':
    # analysis_iuxray()
    # test_chexpert_5_200()
    # data_merge()
    # df_train, chexlabels = get_dataframes('./data/<path>', select= LABEL_SELECTION)
    # df_train, chexlabels = get_dataframes('./data/<path>', select=LABEL_SELECTION)
    # df_valid, _ = get_dataframes('./data/<path>', select= LABEL_SELECTION)
    ## test
    # WRITETYPE = 'test'
    # train_phase2
    WRITETYPE = 'chexpertplus' # 'IUXRAY'

    LABEL_SELECTION = 'chxepertplus' # 'iuxray'  # 'chexBert'
    DF_TYPE = 'available' # 'chexbert_full_zero_or_blank_or_uncertain'
    LABEL_TYPE = 'U-one' # 'U-one'
    # WRITEPATH = './data/<path>'

    WRITEPATH = os.environ.get('WRITEPATH', str(Path(os.environ.get('DATA_ROOT','./data'))/'evals'))
    # WRITEPATH = './data/<path>'

    if not os.path.exists(WRITEPATH):
        os.makedirs(WRITEPATH)
    else:
        print(WRITEPATH, 'already exists')

    # df, chexlabels = get_dataframes_test('./data/<path>', select=LABEL_SELECTION)
    # df_test, chexlabels = get_dataframes_test('./data/<path>',select=LABEL_SELECTION)
    df_test, chexlabels = get_dataframes_test(str(Path(__import__('os').environ.get('DATA_ROOT', './data'))), select=LABEL_SELECTION)
    # df_test, chexlabels = get_dataframes_test('./data/<path>', select=LABEL_SELECTION)

    # df, chexlabels = get_dataframes('./data/<path>',select=LABEL_SELECTION)

    # df_test = df[df[DF_TYPE] == 'no'] # 'chexbert_full_zero_or_blank_or_uncertain-1'
    # df_test = df[df[DF_TYPE] == 'yes']
    # print(f'origin samples:{len(df)}, samples with at least one {LABEL_SELECTION} labeled as 1: {len(df_test)}', )
    print(f'samples:{len(df_test)}')
    test_statics = label_statistics(df_test, chexlabels, 'test')

    # train_dataset = Chexpert_Dataset(df_train, chexlabels, loss_weighting='log', use_augs=True)
    # valid_dataset = Chexpert_Dataset(df_valid, chexlabels)
    test_dataset = Chexpert_Dataset_test(df_test, chexlabels, split='test', root_dir=str(Path(__import__('os').environ.get('DATA_ROOT', './data'))), label_type=LABEL_TYPE)

    # 带权
    # class_weights = torch.tensor(train_dataset.get_class_weights(), dtype=torch.float32)

    class_weights = None

    # train_dataloader, val_dataloader, _ = get_data_loaders(train_dataset, valid_dataset, test_dataset=None)
    testdataloader = get_test_loader(test_dataset)
    # train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    # val_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    # test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    num_classes = len(chexlabels)
    class_names = chexlabels # _x为chexPert _y为chexBert

    # Loss with class weights
    if class_weights is None:
        criterion = BCEWithLogitsLoss()
    else:
        criterion = BCEWithLogitsLoss(pos_weight=class_weights)

    model = ChexpertClassifier(num_classes=num_classes)

    # model = build_model_swintransformer()

    # model = ChexpertClassifier_(num_classes=num_classes)

    # path_to_best_image_classifier_weights = './data/<path>' # chexbert
    # path_to_best_image_classifier_weights = './data/<path>'  # chexpert

    # path_to_best_image_classifier_weights = './data/<path>' # chexbert + uones 15w train best model
    path_to_best_image_classifier_weights = str(Path(__import__('os').environ.get('DATA_ROOT', './data')))



    # swintransformer
    # path_to_best_image_classifier_weights = './data/<path>'
    # resnet50
    # path_to_best_image_classifier_weights = './data/<path>'
    # path_to_best_image_classifier_weights_chexpertplus = './data/<path>'
    load_model = torch.load(path_to_best_image_classifier_weights)
    model.load_state_dict(load_model)
    print('Loaded weights from {}'.format(path_to_best_image_classifier_weights))

    model.to(device, non_blocking=True)
    # model.train()

    # weights_folder_path, tensorboard_folder_path, config_file_path = create_run_folder()
    # writer = SummaryWriter(log_dir=tensorboard_folder_path)
    # opt = AdamW(model.parameters(), lr=LR)
    # scaler = torch.cuda.amp.GradScaler()
    # lr_scheduler = ReduceLROnPlateau(opt, mode="min", factor=FACTOR_LR_SCHEDULER, patience=PATIENCE_LR_SCHEDULER, threshold= THRESHOLD_LR_SCHEDULER, cooldown=COOLDOWN_LR_SCHEDULER)
    # writer = SummaryWriter(log_dir=tensorboard_folder_path)
    print("\nStarting testing!\n")

    test_Img_classifier(model = model,
                        test_dl=testdataloader,
                        label_names = class_names,
                        criterion = criterion,
                        data_origin=df_test,
                        write_path = WRITEPATH,
                        write_type=WRITETYPE,
                        label_selection = LABEL_SELECTION,
                        label_type = LABEL_TYPE)