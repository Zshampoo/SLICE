from pathlib import Path
import shutil
import sys
from ast import literal_eval
from copy import deepcopy
import logging
import os
import random
from typing import List, Dict

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset.constants import ANATOMICAL_REGIONS
from src.object_detector.custom_image_dataset_object_detector import CustomImageDataset, CustomImageDataset_RegionAbnormality
# from src.object_detector.object_detector import ObjectDetector
from region_abnormality_classifier import RegionAbnormalityClassifier
from src.path_datasets_and_weights import path_full_dataset, path_runs_object_detector
import ast
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import torchmetrics
logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
log = logging.getLogger(__name__)

# define configurations for training run
RUN = 'test_demo' # "511_region_abnormality_cls"
# comment can be useful to add additional information to run_config.txt file
RUN_COMMENT = """15w train (phase1+phase2)."""
SEED = 42
IMAGE_INPUT_SIZE = 512
PERCENTAGE_OF_TRAIN_SET_TO_USE = 1.0
PERCENTAGE_OF_VAL_SET_TO_USE = 1.0 #0.2
BATCH_SIZE = 64 #16
EFFECTIVE_BATCH_SIZE = 64
NUM_WORKERS = 10
EPOCHS = 20
LR = 5e-5
EVALUATE_EVERY_K_STEPS = 500  # how often to evaluate the model on the validation set and log metrics to tensorboard (additionally, model will always be evaluated at end of epoch)
PATIENCE_LR_SCHEDULER = 10  # number of evaluations to wait for val loss to reduce before lr is reduced
THRESHOLD_LR_SCHEDULER = 5e-5
FACTOR_LR_SCHEDULER = 0.5
COOLDOWN_LR_SCHEDULER = 5
WEIGHT_OBJECT_DETECTOR_LOSS = 1
# WEIGHT_BINARY_CLASSIFIER_REGION_SELECTION_LOSS = 5
WEIGHT_BINARY_CLASSIFIER_REGION_ABNORMAL_LOSS = 5
# WEIGHT_LANGUAGE_MODEL_LOSS = 2
# set the seed value for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def update_region_abnormal_metrics(region_abnormal_scores, predicted_abnormal_regions, predicted_abnormal_regions_logits, region_is_abnormal, class_detected):
    """
    Args:
        region_abnormal_scores (Dict)
        predicted_abnormal_regions (Tensor[bool]): shape [batch_size x 29]
        region_is_abnormal (Tensor[bool]): shape [batch_size x 29]
        class_detected (Tensor[bool]): shape [batch_size x 29]

    We only update/compute the scores for regions that were actually detected by the object detector (specified by class_detected).
    """
    detected_predicted_abnormal_regions = predicted_abnormal_regions[class_detected]
    detected_predicted_abnormal_regions_logits = predicted_abnormal_regions_logits[class_detected]
    detected_region_is_abnormal = region_is_abnormal[class_detected]


    region_abnormal_scores["precision"](detected_predicted_abnormal_regions, detected_region_is_abnormal)
    region_abnormal_scores["recall"](detected_predicted_abnormal_regions, detected_region_is_abnormal)
    region_abnormal_scores["f1"](detected_predicted_abnormal_regions, detected_region_is_abnormal)
    region_abnormal_scores["acc"](detected_predicted_abnormal_regions.flatten(), detected_region_is_abnormal.flatten())
    region_abnormal_scores["auc"](detected_predicted_abnormal_regions_logits.flatten(), detected_region_is_abnormal.flatten())

def update_object_detector_metrics(obj_detector_scores, detections, image_targets, class_detected):
    def compute_box_area(box):
        """
        Calculate the area of a box given the 4 corner values.

        Args:
            box (Tensor[batch_size x 29 x 4])

        Returns:
            area (Tensor[batch_size x 29])
        """
        x0 = box[..., 0]
        y0 = box[..., 1]
        x1 = box[..., 2]
        y1 = box[..., 3]

        return (x1 - x0) * (y1 - y0)

    def compute_intersection_and_union_area_per_region(detections, targets, class_detected):
        # pred_boxes is of shape [batch_size x 29 x 4] and contains the predicted region boxes with the highest score (i.e. top-1)
        # they are sorted in the 2nd dimension, meaning the 1st of the 29 boxes corresponds to the 1st region/class,
        # the 2nd to the 2nd class and so on
        pred_boxes = detections["top_region_boxes"]

        # targets is a list of dicts, with each dict containing the key "boxes" that contain the gt boxes of a single image
        # gt_boxes is of shape [batch_size x 29 x 4]
        gt_boxes = torch.stack([t["boxes"] for t in targets], dim=0)

        # below tensors are of shape [batch_size x 29]
        x0_max = torch.maximum(pred_boxes[..., 0], gt_boxes[..., 0])
        y0_max = torch.maximum(pred_boxes[..., 1], gt_boxes[..., 1])
        x1_min = torch.minimum(pred_boxes[..., 2], gt_boxes[..., 2])
        y1_min = torch.minimum(pred_boxes[..., 3], gt_boxes[..., 3])

        # intersection_boxes is of shape [batch_size x 29 x 4]
        intersection_boxes = torch.stack([x0_max, y0_max, x1_min, y1_min], dim=-1)

        # below tensors are of shape [batch_size x 29]
        intersection_area = compute_box_area(intersection_boxes)
        pred_area = compute_box_area(pred_boxes)
        gt_area = compute_box_area(gt_boxes)

        # if x0_max >= x1_min or y0_max >= y1_min, then there is no intersection
        valid_intersection = torch.logical_and(x0_max < x1_min, y0_max < y1_min)

        # also there is no intersection if the class was not detected by object detector
        valid_intersection = torch.logical_and(valid_intersection, class_detected)

        # set all non-valid intersection areas to 0
        intersection_area[~valid_intersection] = 0

        union_area = (pred_area + gt_area) - intersection_area

        # sum up the values along the batch dimension (the values will divided by each other later to get the averages)
        intersection_area = torch.sum(intersection_area, dim=0)
        union_area = torch.sum(union_area, dim=0)

        return intersection_area, union_area

    # sum up detections for each region
    region_detected_batch = torch.sum(class_detected, dim=0)

    intersection_area_per_region_batch, union_area_per_region_batch = compute_intersection_and_union_area_per_region(detections, image_targets, class_detected)

    obj_detector_scores["sum_region_detected"] += region_detected_batch
    obj_detector_scores["sum_intersection_area_per_region"] += intersection_area_per_region_batch
    obj_detector_scores["sum_union_area_per_region"] += union_area_per_region_batch

def get_val_losses_and_evaluate_obj_detector_and_binary_classifiers(model, val_dl, log_file, epoch):
    """
    Args:
        model (nn.Module): The input model to be evaluated.
        val_dl (torch.utils.data.Dataloder): The val dataloader to evaluate on.

    Returns:
        val_losses_dict (Dict): holds different val losses of the different modules as well as the total val loss
        obj_detector_scores (Dict): holds scores of the average IoU per Region, average number of detected regions per image,
        average number each region is detected in an image
        region_selection_scores (Dict): holds precision and recall scores for all, normal and abnormal sentences
        region_abnormal_scores (Dict): holds precision and recall scores for all sentences
    """
    model.eval()
    val_losses_dict = {
        "total_loss": 0.0,
        "obj_detector_loss": 0.0,
        "region_abnormal_loss": 0.0,
    }

    # if not PRETRAIN_WITHOUT_LM_MODEL:
    #     val_losses_dict["language_model_loss"] = 0.0

    """
    For the object detector, besides the obj_detector_val_loss, we also want to compute:
      - the average IoU for each region,
      - average number of detected regions per image (ideally 29.0)
      - average number each region is detected in an image (ideally 1.0 for all regions)

    To compute these metrics, we allocate several tensors:

    sum_intersection_area_per_region: for accumulating the intersection area of each region
    (will be divided by union area of each region at the end of get the IoU for each region)

    sum_union_area_per_region: for accumulating the union area of each region
    (will divide the intersection area of each region at the end of get the IoU for each region)

    sum_region_detected: for accumulating the number of times a region is detected over all images
    (this 1D array will be divided by num_images to get the average number each region is detected in an image,
    and these averages will be summed up to get the average number of detected regions in an image)
    """
    obj_detector_scores = {}
    obj_detector_scores["sum_intersection_area_per_region"] = torch.zeros(29, device=device)
    obj_detector_scores["sum_union_area_per_region"] = torch.zeros(29, device=device)
    obj_detector_scores["sum_region_detected"] = torch.zeros(29, device=device)

    # """
    # For the binary classifier for region selection, we want to compute the precision, recall and f1 for:
    #   - all regions
    #   - normal regions
    #   - abnormal regions
    #
    # Evaluation according to:
    #   TP: (normal/abnormal) region has sentence (gt), and is selected by classifier to get sentence (pred)
    #   FP: (normal/abnormal) region does not have sentence (gt), but is selected by classifier to get sentence (pred)
    #   TN: (normal/abnormal) region does not have sentence (gt), and is not selected by classifier to get sentence (pred)
    #   FN: (normal/abnormal) region has sentence (gt), but is not selected by classifier to get sentence (pred)
    # """
    # region_selection_scores = {}
    # for subset in ["all", "normal", "abnormal"]:
    #     region_selection_scores[subset] = {
    #         # specifying average=None computes the metric for each class (i.e. negative and positive) separately
    #         # we then report the score of the positive class by indexing [1] once we've computed the final scores
    #         # this is equivalent to using average="binary" in sklearn.metric (with pos_label=1)
    #         #
    #         # note: using average="micro" is not correct, since it considers the negative and positive classes
    #         # to be separate classes (even in the binary case). If e.g. pred = True and ground-truth = False,
    #         # then it will be considered a FP for the positive class, but also a FN for the negative class,
    #         # which does not make any sense for the binary case and leads to incorrect scores
    #         "precision": torchmetrics.Precision(num_classes=2, average=None).to(device),
    #         "recall": torchmetrics.Recall(num_classes=2, average=None).to(device),
    #         "f1": torchmetrics.F1Score(num_classes=2, average=None).to(device),
    #     }

    """
    For the binary classifier for region normal/abnormal detection, we want to compute the precision, recall and f1 for:
      - all regions

    Evaluation according to:
      TP: region is abnormal (gt), and is predicted as abnormal by classifier (pred)
      FP: region is normal (gt), but is predicted as abnormal by classifier (pred)
      TN: region is normal (gt), and is predicted as normal by classifier (pred)
      FN: region is abnormal (gt), but is predicted as normal by classifier (pred)
    """
    region_abnormal_scores = {
        # specifying average=None computes the metric for each class (i.e. negative and positive) separately
        # we then report the score of the positive class by indexing [1] once we've computed the final scores
        # this is equivalent to using average="binary" in sklearn.metric (with pos_label=1)
        "precision": torchmetrics.classification.MulticlassPrecision(num_classes=2, average=None).to(device),
        "recall": torchmetrics.classification.MulticlassRecall(num_classes=2, average=None).to(device),
        "f1": torchmetrics.classification.MulticlassF1Score(num_classes=2, average=None).to(device),
        "acc": torchmetrics.classification.MulticlassAccuracy(num_classes=2, average=None).to(device),
        "auc": torchmetrics.classification.BinaryAUROC().to(device),
    }

    # to recover from out of memory error if a batch has a sequence that is too long
    oom = False

    num_images = 0

    # for normalizing the val losses
    steps_taken = 0

    with torch.no_grad():
        for num_batch, batch in tqdm(enumerate(val_dl), disable=True):
            ## break
            # if num_batch > 10:
            #     break
            images = batch["images"]
            targets = batch["targets"]
            # region_has_sentence = batch["region_has_sentence"]
            region_is_abnormal = batch["region_is_abnormal"]

            batch_size = images.size(0)
            num_images += batch_size

            images = images.to(device, non_blocking=True)
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
            # region_has_sentence = region_has_sentence.to(device, non_blocking=True)
            region_is_abnormal = region_is_abnormal.to(device, non_blocking=True)

            # if not PRETRAIN_WITHOUT_LM_MODEL:
            #     input_ids = batch["input_ids"]
            #     attention_mask = batch["attention_mask"]
            #
            #     input_ids = input_ids.to(device, non_blocking=True)
            #     attention_mask = attention_mask.to(device, non_blocking=True)
            # else:
            #     input_ids = None
            #     attention_mask = None

            try:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    output = model(images, targets, region_is_abnormal)
            except RuntimeError as e:  # out of memory error
                if "out of memory" in str(e):
                    oom = True

                    with open(log_file, "a") as f:
                        f.write("Evaluation:\n")
                        f.write(f"OOM at epoch {epoch}, batch number {num_batch}.\n")
                        f.write(f"Error message: {str(e)}\n\n")
                else:
                    raise e

            if oom:
                # free up memory
                torch.cuda.empty_cache()
                oom = False

                num_images -= batch_size

                continue

            # output == -1 if the region features that would have been passed into the language model were empty (see forward method for more details)
            # if output == -1:
            #     with open(log_file, "a") as f:
            #         f.write("Evaluation:\n")
            #         f.write(f"Empty region features before language model at epoch {epoch}, batch number {num_batch}.\n\n")
            #
            #     num_images -= batch_size
            #
            #     continue

            # if PRETRAIN_WITHOUT_LM_MODEL:
            #     (
            #         obj_detector_loss_dict,
            #         classifier_loss_region_selection,
            #         classifier_loss_region_abnormal,
            #         detections,
            #         class_detected,
            #         selected_regions,
            #         predicted_abnormal_regions,
            #     ) = output
            # else:
            (   obj_detector_loss_dict,
                # classifier_loss_region_selection,
                classifier_loss_region_abnormal,
                # language_model_loss,
                detections,
                class_detected,  # bool tensor of shape [batch_size x 29]
                # selected_regions,  # bool tensor of shape [batch_size x 29]
                predicted_abnormal_regions,  # bool tensor of shape [batch_size x 29]
                predicted_abnormal_regions_logits
            ) = output

            # detections is a dict with keys "top_region_boxes" and "top_scores"
            # "top_region_boxes" maps to a tensor of shape [batch_size x 29 x 4]
            # "top_scores" maps to a tensor of shape [batch_size x 29]

            # sum up all 4 losses from the object detector
            obj_detector_losses = sum(loss for loss in obj_detector_loss_dict.values())

            # sum up the rest of the losses
            total_loss = WEIGHT_OBJECT_DETECTOR_LOSS * obj_detector_losses + WEIGHT_BINARY_CLASSIFIER_REGION_ABNORMAL_LOSS * classifier_loss_region_abnormal

            # if not PRETRAIN_WITHOUT_LM_MODEL:
            #     total_loss += WEIGHT_LANGUAGE_MODEL_LOSS * language_model_loss

            list_of_losses = [
                total_loss,
                obj_detector_losses,
                classifier_loss_region_abnormal,
            ]

            # if not PRETRAIN_WITHOUT_LM_MODEL:
            #     list_of_losses.append(language_model_loss)

            # dicts are insertion ordered since Python 3.7
            for loss_type, loss in zip(val_losses_dict, list_of_losses):
                val_losses_dict[loss_type] += loss.item() * batch_size

            steps_taken += 1

            # update scores for object detector metrics
            update_object_detector_metrics(obj_detector_scores, detections, targets, class_detected)

            # update scores for region selection metrics
            # update_region_selection_metrics(region_selection_scores, selected_regions, region_has_sentence, region_is_abnormal)

            # update scores for region abnormal detection metrics
            update_region_abnormal_metrics(region_abnormal_scores, predicted_abnormal_regions, predicted_abnormal_regions_logits, region_is_abnormal, class_detected)

    # normalize the val losses by steps_taken
    for loss_type in val_losses_dict:
        val_losses_dict[loss_type] /= steps_taken

    # compute object detector scores
    sum_intersection = obj_detector_scores["sum_intersection_area_per_region"]
    sum_union = obj_detector_scores["sum_union_area_per_region"]
    obj_detector_scores["avg_iou"] = (torch.sum(sum_intersection) / torch.sum(sum_union)).item()
    obj_detector_scores["avg_iou_per_region"] = (sum_intersection / sum_union).tolist()

    sum_region_detected = obj_detector_scores["sum_region_detected"]
    obj_detector_scores["avg_num_detected_regions_per_image"] = torch.sum(sum_region_detected / num_images).item()
    obj_detector_scores["avg_detections_per_region"] = (sum_region_detected / num_images).tolist()

    # compute the "micro" average scores for region_selection_scores
    # for subset in region_selection_scores:
    #     for metric, score in region_selection_scores[subset].items():
    #         region_selection_scores[subset][metric] = score.compute()[1].item()  # only report results for the positive class (hence [1])

    # compute the "micro" average scores for region_abnormal_scores
    # for metric, score in region_abnormal_scores.items():
    #     region_abnormal_scores[metric] = score.compute()[1].item()

    for metric, score in region_abnormal_scores.items():
        computed = score.compute()
        if isinstance(computed, torch.Tensor) and computed.numel() == 1:
            region_abnormal_scores[metric] = computed.item()
        else:
            region_abnormal_scores[metric] = computed  # 保持原样，比如是一个张量 [2] 或 [num_classes]

    return val_losses_dict, obj_detector_scores, region_abnormal_scores

def log_stats_to_console(
    train_loss,
    val_loss,
    epoch,
):
    log.info(f"Epoch: {epoch}:")
    log.info(f"\tTrain total loss: {train_loss:.3f}")
    log.info(f"\tVal total loss: {val_loss:.3f}")


def train_model(
    model,
    train_dl,
    val_dl,
    optimizer,
    scaler,
    lr_scheduler,
    epochs,
    weights_folder_path,
    writer
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
    lowest_val_loss = np.inf

    # the best_model_state is the one where the val loss is the lowest overall
    best_model_state = None

    overall_steps_taken = 0  # for logging to tensorboard

    # for gradient accumulation
    ACCUMULATION_STEPS = EFFECTIVE_BATCH_SIZE // BATCH_SIZE

    for epoch in range(epochs):
        log.info(f"Training region_abnormality_classifier epoch {epoch}!")

        train_losses_dict = {
            "total_loss": 0.0,
            "obj_detector_loss": 0.0,
            "region_abnormal_loss": 0.0,
        }
        steps_taken = 0
        Summary = ""
        for num_batch, batch in tqdm(enumerate(train_dl), total=len(train_dl), disable=False):
            # break
            # if num_batch > 200:
            #     break
            # batch is a dict with keys "images" and "targets"

            images = batch["images"]
            targets = batch["targets"]
            region_is_abnormal = batch["region_is_abnormal"]

            batch_size = images.size(0)

            images = images.to(device, non_blocking=True)  # shape (batch_size x 1 x 512 x 512)
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
            region_is_abnormal = region_is_abnormal.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                obj_detector_loss_dict, classifier_loss_region_abnormal = model(images, targets, region_is_abnormal)

                # sum up all 4 losses from the object detector
                obj_detector_losses = sum(loss for loss in obj_detector_loss_dict.values())

                total_loss = (WEIGHT_OBJECT_DETECTOR_LOSS * obj_detector_losses  +
                              WEIGHT_BINARY_CLASSIFIER_REGION_ABNORMAL_LOSS * classifier_loss_region_abnormal)

            scaler.scale(total_loss).backward()

            if (num_batch + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # dicts are insertion ordered since Python 3.7
            list_of_losses = [
                total_loss,
                obj_detector_losses,
                classifier_loss_region_abnormal,
            ]

            for loss_type, loss in zip(train_losses_dict, list_of_losses):
                train_losses_dict[loss_type] += loss.item() * batch_size
            steps_taken += 1
            overall_steps_taken += 1

            # evaluate every k steps and also at the end of an epoch
            if steps_taken >= EVALUATE_EVERY_K_STEPS or (num_batch + 1) == len(train_dl):
                log.info(f"Evaluating at step {overall_steps_taken}!")

                # normalize the train loss by steps_taken
                for loss_type in train_losses_dict:
                    train_losses_dict[loss_type] /= steps_taken

                val_loss_dict, obj_detector_scores, region_abnormal_scores = get_val_losses_and_evaluate_obj_detector_and_binary_classifiers(model, val_dl, writer, overall_steps_taken)

                # loss
                for train_loss_key, train_loss_value in train_losses_dict.items():
                    writer.add_scalar(f"train_loss_{train_loss_key}:", train_loss_value, overall_steps_taken)
                for val_loss_key, val_loss_value in val_loss_dict.items():
                    writer.add_scalar(f"val_loss_{val_loss_key}:", val_loss_value, overall_steps_taken)
                # object detector metrics
                for met_od_key, met_od_value in obj_detector_scores.items():
                    if isinstance(met_od_value, torch.Tensor):
                        continue
                    elif isinstance(met_od_value, list):
                        for index, v in enumerate(met_od_value):
                            writer.add_scalar(f"object_detector_metrics_{met_od_key}_region_{index}:", v, overall_steps_taken)
                    elif isinstance(met_od_value, float):
                        writer.add_scalar(f"object_detector_metrics_{met_od_key}:", met_od_value, overall_steps_taken)
                    else:
                        print(f"error type of {met_od_key}")
                # region abnormality classifier metrics
                for met_ra_key, met_ra_value in region_abnormal_scores.items():
                    if isinstance(met_ra_value, torch.Tensor):
                        writer.add_scalar(f"region_abnormality_classifier_metrics_{met_ra_key}_normal:", met_ra_value[0], overall_steps_taken)
                        writer.add_scalar(f"region_abnormality_classifier_metrics_{met_ra_key}_abnormal:", met_ra_value[1], overall_steps_taken)
                    elif isinstance(met_ra_value, float):
                        writer.add_scalar(f"region_abnormality_classifier_metrics_{met_ra_key}:", met_ra_value, overall_steps_taken)
                    else:
                        print(f"error type of {met_ra_key}")
                current_lr = float(optimizer.param_groups[0]["lr"])
                writer.add_scalar(f"lr:", current_lr, overall_steps_taken)

                log.info(f"Metrics evaluated at step {overall_steps_taken}!")

                # summary
                Summary += (f"\nEpoch [{epoch}][{num_batch}/{len(train_dl)}] Step {overall_steps_taken}:\n")
                for train_loss_key, train_loss_value in train_losses_dict.items():
                    Summary += (f"train_{train_loss_key}:{train_loss_value:.6f}\n")
                for val_loss_key, val_loss_value in val_loss_dict.items():
                    Summary += (f"val_{val_loss_key}:{val_loss_value:.6f}\n")
                Summary += f"\n**********Object Detector Evaluation**********\n"
                for met_od_key, met_od_value in obj_detector_scores.items():
                    if isinstance(met_od_value, torch.Tensor):
                        continue
                    elif isinstance(met_od_value, list):
                        for index, v in enumerate(met_od_value):
                            Summary += (f"{met_od_key}_region_{index}:{v:.6f}\n")
                    elif isinstance(met_od_value, float):
                        Summary += (f"{met_od_key}:{met_od_value:.6f}\n")
                    else:
                        print(f"error type of {met_od_key}")
                Summary += f"\n**********Region Abnormality Classifier Evaluation**********\n"
                # region abnormality classifier metrics
                for met_ra_key, met_ra_value in region_abnormal_scores.items():
                    if isinstance(met_ra_value, torch.Tensor):
                        Summary += (f"{met_ra_key}_normal:{met_ra_value[0]:.6f}, {met_ra_key}_abnormal:{met_ra_value[1]:.6f}\n")
                    elif isinstance(met_ra_value, float):
                        Summary += (f"{met_ra_key}:{met_ra_value:.6f}\n")
                    else:
                        print(f"error type of {met_ra_key}")
                # print
                model.train()

                # decrease lr if val loss has not decreased after certain number of evaluations
                total_val_loss = val_loss_dict["total_loss"]
                lr_scheduler.step(total_val_loss)

                if total_val_loss < lowest_val_loss:
                    lowest_val_loss = total_val_loss
                    best_epoch = epoch
                    avg_iou, avg_detections = obj_detector_scores["avg_iou"] ,obj_detector_scores["avg_num_detected_regions_per_image"]
                    f1, acc, auc = region_abnormal_scores["f1"], region_abnormal_scores["acc"], region_abnormal_scores["auc"]
                    best_model_save_path = os.path.join(
                        weights_folder_path, f"val_loss_{lowest_val_loss:.3f}_epoch_{epoch}_step_{overall_steps_taken}_od_avgIOU_{avg_iou}"
                                             f"_od_avgdetections_{avg_detections}_rac_f1_{f1}_rac_acc_{acc}_rac_auc_{auc}.pth")
                    best_model_state = deepcopy(model.state_dict())

                # log to console at the end of an epoch
                total_train_loss = train_losses_dict["total_loss"]
                if (num_batch + 1) == len(train_dl):
                    log_stats_to_console(train_losses_dict["total_loss"], val_loss_dict["total_loss"], epoch)

                # reset values
                for loss_type in train_losses_dict:
                    train_losses_dict[loss_type] = 0.0
                steps_taken = 0

        # print
        tqdm.write(Summary)
        run_folder_path = os.path.join(path_runs_object_detector, f"run_{RUN}")
        with open(run_folder_path + "/training_summary_epoch{}.txt".format(epoch), "w") as f:
            f.write(Summary)
        # save the current best model weights at the end of each epoch
        torch.save(best_model_state, best_model_save_path)

    log.info("Finished training!")
    log.info(f"Lowest total val loss: {lowest_val_loss:.3f} at epoch {best_epoch}")
    return None


def collate_fn(batch: List[Dict[str, Tensor]]):
    # each dict in batch (which is a list) is for a single image and has the keys "image", "boxes", "labels"

    # discard images from batch where __getitem__ from custom_image_dataset failed (i.e. returned None)
    # otherwise, whole training loop will stop (even if only 1 image fails to open)
    batch = list(filter(lambda x: x is not None, batch))

    image_shape = batch[0]["image"].size()
    # allocate an empty images_batch tensor that will store all images of the batch
    images_batch = torch.empty(size=(len(batch), *image_shape))
    bbox_is_abnormal_size = batch[0]["bbox_is_abnormal"].size()  # should be torch.Size([29])
    region_is_abnormal = torch.empty(size=(len(batch), * bbox_is_abnormal_size), dtype=torch.bool)
    image_targets = []

    for i, sample in enumerate(batch):
        # remove image tensors from batch and store them in dedicated images_batch tensor
        images_batch[i] = sample.pop("image")
        boxes = sample.pop("boxes")
        labels = sample.pop("labels")
        image_targets.append({"boxes": boxes, "labels": labels})
        region_is_abnormal[i] = sample.pop("bbox_is_abnormal")

    # since batch (which is a list) now only contains dicts with keys "boxes" and "labels", rename it as targets
    # targets = batch

    # create a new batch variable to store images_batch and targets
    batch_new = {}
    batch_new["images"] = images_batch
    batch_new["targets"] = image_targets
    batch_new["region_is_abnormal"] = region_is_abnormal

    return batch_new


def get_data_loaders(train_dataset, val_dataset):
    def seed_worker(worker_id):
        """To preserve reproducibility for the randomly shuffled train loader."""
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, worker_init_fn=seed_worker, generator=g, pin_memory=True)
    val_loader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader


def get_transforms(dataset: str):
    # see compute_mean_std_dataset.py in src/dataset
    mean = 0.471
    std = 0.302

    # use albumentations for Compose and transforms
    # augmentations are applied with prob=0.5
    # since Affine translates and rotates the image, we also have to do the same with the bounding boxes, hence the bbox_params arugment
    train_transforms = A.Compose(
        [
            # we want the long edge of the image to be resized to IMAGE_INPUT_SIZE, and the short edge of the image to be padded to IMAGE_INPUT_SIZE on both sides,
            # such that the aspect ratio of the images are kept, while getting images of uniform size (IMAGE_INPUT_SIZE x IMAGE_INPUT_SIZE)
            # LongestMaxSize: resizes the longer edge to IMAGE_INPUT_SIZE while maintaining the aspect ratio
            # INTER_AREA works best for shrinking images
            A.LongestMaxSize(max_size=IMAGE_INPUT_SIZE, interpolation=cv2.INTER_AREA),
            A.ColorJitter(hue=0.0),
            A.GaussNoise(),
            # randomly (by default prob=0.5) translate and rotate image
            # mode and cval specify that black pixels are used to fill in newly created pixels
            # translate between -2% and 2% of the image height/width, rotate between -2 and 2 degrees
            A.Affine(mode=cv2.BORDER_CONSTANT, cval=0, translate_percent=(-0.02, 0.02), rotate=(-2, 2)),
            # PadIfNeeded: pads both sides of the shorter edge with 0's (black pixels)
            A.PadIfNeeded(min_height=IMAGE_INPUT_SIZE, min_width=IMAGE_INPUT_SIZE, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=['class_labels'])
    )

    # don't apply data augmentations to val and test set
    val_test_transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=IMAGE_INPUT_SIZE, interpolation=cv2.INTER_AREA),
            A.PadIfNeeded(min_height=IMAGE_INPUT_SIZE, min_width=IMAGE_INPUT_SIZE, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=['class_labels'])
    )

    if dataset == "train":
        return train_transforms
    else:
        return val_test_transforms


def get_datasets_as_dfs_ra(config_file_path):
    usecols = ["mimic_image_file_path", "bbox_coordinates", "bbox_labels", "bbox_is_abnormal", "all_reference_report"]

    # since bbox_coordinates and bbox_labels are stored as strings in the csv_file, we have to apply
    # the literal_eval func to convert them to python lists
    converters = {"bbox_coordinates": literal_eval, "bbox_labels": literal_eval}

    # datasets_as_dfs = {dataset: os.path.join(path_full_dataset, dataset) + ".xlsx" for dataset in ["train", "valid"]}
    datasets_as_dfs = {}
    datasets_as_dfs['train'] = str(Path(__import__('os').environ.get('DATA_ROOT', './data')))
    datasets_as_dfs['valid'] = str(Path(__import__('os').environ.get('DATA_ROOT', './data')))
    # datasets_as_dfs = {dataset: pd.read_excel(file_path, usecols=usecols, converters=converters,  engine='openpyxl') for dataset, file_path in datasets_as_dfs.items()}
    datasets_as_dfs = {dataset: pd.read_excel(file_path, usecols=usecols, engine='openpyxl') for dataset, file_path in datasets_as_dfs.items()}

    # datasets_as_dfs = {}
    # datasets_as_dfs['train'] = pd.read_excel('./data/<path>', usecols=usecols, engine='openpyxl')
    # datasets_as_dfs['valid'] = pd.read_excel('./data/<path>', usecols=usecols, engine='openpyxl')
    datasets_as_dfs['train']['bbox_coordinates'] = datasets_as_dfs['train']['bbox_coordinates'].apply(ast.literal_eval)
    datasets_as_dfs['train']['bbox_labels'] = datasets_as_dfs['train']['bbox_labels'].apply(ast.literal_eval)
    datasets_as_dfs['train']['bbox_is_abnormal'] = datasets_as_dfs['train']['bbox_is_abnormal'].apply(ast.literal_eval)
    # datasets_as_dfs['train']['all_reference_report'] = datasets_as_dfs['train']['all_reference_report'].apply(ast.literal_eval)

    datasets_as_dfs['valid']['bbox_coordinates'] = datasets_as_dfs['valid']['bbox_coordinates'].apply(ast.literal_eval)
    datasets_as_dfs['valid']['bbox_labels'] = datasets_as_dfs['valid']['bbox_labels'].apply(ast.literal_eval)
    datasets_as_dfs['valid']['bbox_is_abnormal'] = datasets_as_dfs['valid']['bbox_is_abnormal'].apply(ast.literal_eval)
    # datasets_as_dfs['valid']['all_reference_report'] = datasets_as_dfs['valid']['all_reference_report'].apply(ast.literal_eval)

    total_num_samples_train = len(datasets_as_dfs["train"])
    total_num_samples_val = len(datasets_as_dfs["valid"])

    # compute new number of samples for both train and val
    new_num_samples_train = int(PERCENTAGE_OF_TRAIN_SET_TO_USE * total_num_samples_train)
    new_num_samples_val = int(PERCENTAGE_OF_VAL_SET_TO_USE * total_num_samples_val)

    log.info(f"Train: {new_num_samples_train} images")
    log.info(f"Val: {new_num_samples_val} images")

    with open(config_file_path, "a") as f:
        f.write(f"\tTRAIN NUM IMAGES: {new_num_samples_train}\n")
        f.write(f"\tVAL NUM IMAGES: {new_num_samples_val}\n")

    # limit the datasets to those new numbers
    datasets_as_dfs["train"] = datasets_as_dfs["train"][:new_num_samples_train]
    datasets_as_dfs["valid"] = datasets_as_dfs["valid"][:new_num_samples_val]

    return datasets_as_dfs


def create_run_folder():
    """
    Run folder will contain a folder for saving the trained weights, a folder for the tensorboard files
    as well as a config file that specifies the overall parameters used for training.
    """
    run_folder_path = os.path.join(path_runs_object_detector, f"run_{RUN}")
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
        "COMMENT": RUN_COMMENT,
        "SEED": SEED,
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
        "THRESHOLD_LR_SCHEDULER": THRESHOLD_LR_SCHEDULER,
        "FACTOR_LR_SCHEDULER": FACTOR_LR_SCHEDULER,
        "COOLDOWN_LR_SCHEDULER": COOLDOWN_LR_SCHEDULER
    }

    with open(config_file_path, "w") as f:
        f.write(f"RUN {RUN}:\n")
        for param_name, param_value in config_parameters.items():
            f.write(f"\t{param_name}: {param_value}\n")

    return weights_folder_path, tensorboard_folder_path, config_file_path

def normal_percent_statistic(df):
    dict_normal_abnormality = {}
    normality_count, abnormality_count = 0, 0
    for i in range(29):
        dict_normal_abnormality[i] = {'normal': 0, 'abnormal': 0}
    for index, row in df.iterrows():
        bbox_is_normal = row["bbox_is_abnormal"]
        for i in range(len(bbox_is_normal)):
            if bbox_is_normal[i]:
                dict_normal_abnormality[i]['abnormal'] += 1
            else:
                dict_normal_abnormality[i]['normal'] += 1
    for i in range(29):
        normality_count += dict_normal_abnormality[i]['normal'] # train:2808310 val:149534
        abnormality_count += dict_normal_abnormality[i]['abnormal'] # train:437834 (6.4:1) val:23248(6.4:1)
    return dict_normal_abnormality


def main():
    weights_folder_path, tensorboard_folder_path, config_file_path = create_run_folder()

    datasets_as_dfs = get_datasets_as_dfs_ra(config_file_path)

    train_transforms = get_transforms("train")
    val_transforms = get_transforms("val")

    # statistical
    dict_normal_abnormality_train = normal_percent_statistic(datasets_as_dfs["train"])
    dict_normal_abnormality_val = normal_percent_statistic(datasets_as_dfs["valid"])

    train_dataset = CustomImageDataset_RegionAbnormality(datasets_as_dfs["train"], train_transforms)
    val_dataset = CustomImageDataset_RegionAbnormality(datasets_as_dfs["valid"], val_transforms)

    train_loader, val_loader = get_data_loaders(train_dataset, val_dataset)

    # path_to_best_object_detector_weights = './data/<path>'
    path_to_best_object_detector_weights = str(Path(__import__('os').environ.get('DATA_ROOT', './data')))

    model = RegionAbnormalityClassifier(path_to_best_object_detector_weights) # TODO
    model.to(device, non_blocking=True)
    model.train()

    scaler = torch.cuda.amp.GradScaler()

    opt = AdamW(model.parameters(), lr=LR)
    lr_scheduler = ReduceLROnPlateau(opt, mode="min", factor=FACTOR_LR_SCHEDULER, patience=PATIENCE_LR_SCHEDULER, threshold=THRESHOLD_LR_SCHEDULER, cooldown=COOLDOWN_LR_SCHEDULER)
    writer = SummaryWriter(log_dir=tensorboard_folder_path)
    log.info("\nStarting training!\n")

    train_model(
        model=model,
        train_dl=train_loader,
        val_dl=val_loader,
        optimizer=opt,
        scaler=scaler,
        lr_scheduler=lr_scheduler,
        epochs=EPOCHS,
        weights_folder_path=weights_folder_path,
        writer=writer
    )


if __name__ == "__main__":
    # 201M
    # model = RegionAbnormalityClassifier()  # TODO
    # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Number of params: {n_parameters}")
    # print(f"Number of params: {n_parameters:,} ({n_parameters / 1e6:.2f}M)")
    # since bbox_coordinates and bbox_labels are stored as strings in the csv_file, we have to apply
    # the literal_eval func to convert them to python lists
    # usecols = ["mimic_image_file_path", "bbox_coordinates", "bbox_labels"]
    # converters = {"bbox_coordinates": literal_eval, "bbox_labels": literal_eval}
    # # datasets_as_dfs = pd.read_excel('./data/<path>', usecols=usecols, engine='openpyxl')
    # # datasets_as_dfs_train = pd.read_excel('./data/<path>', usecols=usecols, engine='openpyxl')
    # datasets_as_dfs_valid = pd.read_excel('./data/<path>', usecols=usecols, engine='openpyxl')
    # datasets_as_dfs_valid['bbox_coordinates'] = datasets_as_dfs_valid['bbox_coordinates'].apply(ast.literal_eval)
    # print(torch.__version__)  # PyTorch版本
    # print(torch.version.cuda)  # PyTorch使用的CUDA版本
    # print(torch.cuda.get_device_name(0))  # 你的GPU名字
    main()
