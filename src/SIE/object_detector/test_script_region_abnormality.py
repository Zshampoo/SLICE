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
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset.constants import ANATOMICAL_REGIONS
from src.SIE.object_detector.custom_image_dataset_object_detector import CustomImageDataset_RegionAbnormality_test
# from src.SIE.object_detector.object_detector import ObjectDetector
from region_abnormality_classifier import RegionAbnormalityClassifier
from src.path_datasets_and_weights import path_full_dataset, path_runs_object_detector
import ast
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import torchmetrics
logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
log = logging.getLogger(__name__)

ANATOMICAL_REGIONS = {
    "right lung": 0,
    "right upper lung zone": 1,
    "right mid lung zone": 2,
    "right lower lung zone": 3,
    "right hilar structures": 4,
    "right apical zone": 5,
    "right costophrenic angle": 6,
    "right hemidiaphragm": 7,
    "left lung": 8,
    "left upper lung zone": 9,
    "left mid lung zone": 10,
    "left lower lung zone": 11,
    "left hilar structures": 12,
    "left apical zone": 13,
    "left costophrenic angle": 14,
    "left hemidiaphragm": 15,
    "trachea": 16,
    "spine": 17,
    "right clavicle": 18,
    "left clavicle": 19,
    "aortic arch": 20,
    "mediastinum": 21,
    "upper mediastinum": 22,
    "svc": 23,
    "cardiac silhouette": 24,
    "cavoatrial junction": 25,
    "right atrium": 26,
    "carina": 27,
    "abdomen": 28
}
# define configurations for training run
RUN = "chexpertplus_zeroshot_region_abnormality"
# comment can be useful to add additional information to run_config.txt file
SEED = 43
IMAGE_INPUT_SIZE = 512
# PERCENTAGE_OF_TRAIN_SET_TO_USE = 1.0
# PERCENTAGE_OF_VAL_SET_TO_USE = 1.0 #0.2
BATCH_SIZE = 16 #16
EFFECTIVE_BATCH_SIZE = 64
NUM_WORKERS = 10
# EPOCHS = 20
# LR = 5e-5
# EVALUATE_EVERY_K_STEPS = 500  # how often to evaluate the model on the validation set and log metrics to tensorboard (additionally, model will always be evaluated at end of epoch)
# PATIENCE_LR_SCHEDULER = 10  # number of evaluations to wait for val loss to reduce before lr is reduced
# THRESHOLD_LR_SCHEDULER = 1e-3
# FACTOR_LR_SCHEDULER = 0.5
# COOLDOWN_LR_SCHEDULER = 5
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

def test_model_zeroshot(model, test_dl, writepath_root, data_origin, writetype='test'):
    """
    Zero-shot inference on datasets without region-level GT (e.g. CheXpert+).
    Skips IoU / ACC / F1 / AUC that require GT; only saves per-sample predictions
    and detection-rate summary (class_detected counts).
    """
    log.info("Zero-shot region_abnormality_classifier (no region GT metrics)!")
    sum_region_detected = torch.zeros(29, device=device)
    num_images = 0
    result_tuples = []
    with torch.no_grad():
        model.eval()
        for num_batch, batch in tqdm(enumerate(test_dl), leave=False, total=len(test_dl)):
            images = batch["images"].to(device, non_blocking=True)
            file_id = batch["file_id"]
            batch_size = images.size(0)
            num_images += batch_size

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                # targets=None: no GT boxes; detector returns detections + region features
                _, detections, top_region_features, class_detected = model.object_detector(images, None)
                logits = model.binary_classifier_region_abnormal.classifier(top_region_features).squeeze(dim=-1)
                predicted_abnormal_regions = logits > -1

            sum_region_detected += torch.sum(class_detected, dim=0)
            for i in range(batch_size):
                result_tuples.append([
                    file_id[i],
                    detections["top_region_boxes"][i].detach().cpu().numpy(),
                    class_detected[i].detach().cpu().numpy(),
                    predicted_abnormal_regions[i].detach().cpu().numpy(),
                    logits[i].detach().cpu().numpy(),
                ])

    avg_det_per_image = torch.sum(sum_region_detected / num_images).item()
    avg_det_per_region = (sum_region_detected / num_images).tolist()
    summary_path = os.path.join(writepath_root, f"{writetype}_zeroshot_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Test sample count: {num_images}\n")
        f.write("(No region-level GT: skipped IoU / ACC / F1 / AUC)\n")
        f.write(f"AVG_NUM_DETECTED_REGIONS_PER_IMAGE: {avg_det_per_image}\n")
        for i in range(29):
            f.write(f"AVG_DETECTIONS_REGION_{i}: {avg_det_per_region[i]}\n")
        # prediction prevalence (among detected regions)
        pred_abn = 0
        det = 0
        for _, _, cd, pred, _ in result_tuples:
            cd_b = cd.astype(bool)
            det += int(cd_b.sum())
            pred_abn += int((pred.astype(bool) & cd_b).sum())
        f.write(f"Detected regions total: {det}\n")
        f.write(f"Predicted abnormal among detected: {pred_abn} ({100.0 * pred_abn / max(det, 1):.2f}%)\n")
    log.info(f"Wrote summary to {summary_path}")

    persample_path = save_persample_result(data_origin, result_tuples, writepath_root, writetype)
    log.info(f"Wrote per-sample results to {persample_path}")
    log.info("Finished zero-shot test!")
    return summary_path, persample_path

def save_overall_scores(obj_detector_scores, region_abnormal_scores, writepath_root, writetype='test'):
    od_file_path = writepath_root + '/{}_result.txt'.format(writetype)
    with open(od_file_path, 'w') as f:
        f.write('Test sample count:' + str(obj_detector_scores["total_test_sample"]) + '\n')
        f.write('*************** Object Detector Scores ***************\n')
        f.write('AVG_IOU:' + str(obj_detector_scores["avg_iou"]) + '\n')
        f.write('AVG_NUM_DETECTED_REGIONS_PER_IMAGE:' + str(obj_detector_scores["avg_num_detected_regions_per_image"]) + '\n')
        for i in range(29):
            f.write('AVG_IOU_REGION_{}:'.format(i) +str(obj_detector_scores["avg_iou_per_region"][i]) + '\n')
        for i in range(29):
            f.write('AVG_DETECTIONS_REGION_{}:'.format(i) + str(obj_detector_scores["avg_detections_per_region"][i]) + '\n')
        f.write('*************** Region Abnormality Classification Scores ***************\n')
        f.write(f'ACC_normal: {region_abnormal_scores["acc"][0]:.6f} ACC_abnormal: {region_abnormal_scores["acc"][1]:.6f}\n')
        f.write(f'AUC: {region_abnormal_scores["auc"]:.6f}\n')
        f.write(f'F1_normal: {region_abnormal_scores["f1"][0]:.6f} F1_abnormal: {region_abnormal_scores["f1"][1]:.6f}\n')
        f.write(f'Precision_normal: {region_abnormal_scores["precision"][0]:.6f} Precision_abnormal: {region_abnormal_scores["precision"][1]:.6f}\n')
        f.write(f'Recall_normal: {region_abnormal_scores["recall"][0]:.6f} Recall_abnormal: {region_abnormal_scores["recall"][1]:.6f}\n')
    return od_file_path

def save_overall_scores_micro(obj_detector_scores, region_abnormal_scores, writepath_root, writetype='test'):
    od_file_path = writepath_root + '/{}_result_micro.txt'.format(writetype)
    with open(od_file_path, 'w') as f:
        f.write('Test sample count:' + str(obj_detector_scores["total_test_sample"]) + '\n')
        f.write('*************** Object Detector Scores ***************\n')
        f.write('AVG_IOU:' + str(obj_detector_scores["avg_iou"]) + '\n')
        f.write('AVG_NUM_DETECTED_REGIONS_PER_IMAGE:' + str(obj_detector_scores["avg_num_detected_regions_per_image"]) + '\n')
        for i in range(29):
            f.write('AVG_IOU_REGION_{}:'.format(i) +str(obj_detector_scores["avg_iou_per_region"][i]) + '\n')
        for i in range(29):
            f.write('AVG_DETECTIONS_REGION_{}:'.format(i) + str(obj_detector_scores["avg_detections_per_region"][i]) + '\n')
        f.write('*************** Region Abnormality Classification Scores ***************\n')
        f.write(f'ACC_micro: {region_abnormal_scores["acc"]:.6f} \n')
        f.write(f'AUC_micro: {region_abnormal_scores["auc"]:.6f}\n')
        f.write(f'F1_micro: {region_abnormal_scores["f1"]:.6f} \n')
        f.write(f'Precision_micro: {region_abnormal_scores["precision"]:.6f}\n')
        f.write(f'Recall_micro: {region_abnormal_scores["recall"]:.6f} \n')
    return od_file_path

def save_persample_result(df, result, writepath_root, writetype='test'):
    # assert len(df) == len(result), "shape not same!"
    df_result = pd.DataFrame(result, columns=['mimic_image_file_path',  'top_region_boxes', 'class_detected', 'predicted_abnormal_regions', 'predicted_abnormal_regions_logits']) #["mimic_image_file_path", "bbox_coordinates", "bbox_labels", "bbox_is_abnormal", "all_reference_report"]
    # logit
    logits_array = np.stack(df_result['predicted_abnormal_regions_logits'].values)
    logits_tensor = torch.tensor(logits_array, dtype=torch.float32)
    sigmoid = nn.Sigmoid()
    probs = list(sigmoid(logits_tensor).numpy()) # 对应于正类（label=1， abnormal）的置信度分数
    df_result['predicted_abnormal_regions_probs'] = probs
    # combination
    pred_result = df_result['predicted_abnormal_regions'].values.tolist()
    detected_result = df_result['class_detected'].values.tolist()
    result2ab = {True:'Abnormal', False:'Normal'}
    pred_region_info_tuple = []
    pred_region_info_only_detected_tuple = []
    undetected_region_names_list = []
    for i in range(len(pred_result)):
        sample_tuple = []
        sample_tuple_detected = []
        undetected_names = []
        for key in ANATOMICAL_REGIONS.keys():
            no = ANATOMICAL_REGIONS[key]
            pred = result2ab[bool(pred_result[i][no])]
            prob = probs[i][no]
            conf = prob if pred == 'Abnormal' else (1 - prob)
            sample_tuple.append([key, pred, conf])
            # 仅保留检出区域；未检出区域名单独列记录
            if not bool(detected_result[i][no]):
                undetected_names.append(key)
            else:
                sample_tuple_detected.append([key, pred, conf])
        pred_region_info_tuple.append(sample_tuple)
        pred_region_info_only_detected_tuple.append(sample_tuple_detected)
        undetected_region_names_list.append(undetected_names)
    df_result['region_info'] = pred_region_info_tuple
    df_result['region_info_only_detected'] = pred_region_info_only_detected_tuple
    df_result['undetected_regions'] = undetected_region_names_list

    df_new = pd.merge(df, df_result, on='mimic_image_file_path', how='outer')
    df_new = df_new.astype(str)

    path = writepath_root + '/persample_{}_result.xlsx'.format(writetype)
    df_cleaned = df_new.applymap(lambda x: x.replace('\n', ' ') if isinstance(x, str) else x)
    df_cleaned.to_excel(path, index=False)
    return path

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
    file_ids = []

    for i, sample in enumerate(batch):
        # remove image tensors from batch and store them in dedicated images_batch tensor
        images_batch[i] = sample.pop("image")
        boxes = sample.pop("boxes")
        labels = sample.pop("labels")
        image_targets.append({"boxes": boxes, "labels": labels})
        region_is_abnormal[i] = sample.pop("bbox_is_abnormal")
        file_ids.append(sample.pop("file_id"))

    # since batch (which is a list) now only contains dicts with keys "boxes" and "labels", rename it as targets
    # targets = batch

    # create a new batch variable to store images_batch and targets
    batch_new = {}
    batch_new["images"] = images_batch
    batch_new["targets"] = image_targets
    batch_new["region_is_abnormal"] = region_is_abnormal
    batch_new["file_id"] = file_ids

    return batch_new


def get_data_loaders(test_dataset):
    def seed_worker(worker_id):
        """To preserve reproducibility for the randomly shuffled train loader."""
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(SEED)

    test_loader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    return test_loader


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


def get_datasets_as_dfs_test(path_test_dataset):
    usecols = ["mimic_image_file_path", "bbox_coordinates", "bbox_labels", "bbox_is_abnormal", "all_reference_report", "available"]

    # since bbox_coordinates and bbox_labels are stored as strings in the csv_file, we have to apply
    # the literal_eval func to convert them to python lists
    converters = {"bbox_coordinates": literal_eval, "bbox_labels": literal_eval}
    datasets_as_dfs = pd.read_excel(path_test_dataset, usecols=usecols, engine='openpyxl')
    # datasets_as_dfs = {}
    # datasets_as_dfs['train'] = pd.read_excel('./data/<path>', usecols=usecols, engine='openpyxl')
    # datasets_as_dfs['valid'] = pd.read_excel('./data/<path>', usecols=usecols, engine='openpyxl')
    datasets_as_dfs['bbox_coordinates'] = datasets_as_dfs['bbox_coordinates'].apply(ast.literal_eval)
    datasets_as_dfs['bbox_labels'] = datasets_as_dfs['bbox_labels'].apply(ast.literal_eval)
    datasets_as_dfs['bbox_is_abnormal'] = datasets_as_dfs['bbox_is_abnormal'].apply(ast.literal_eval)

    total_num_samples_test = len(datasets_as_dfs)

    log.info(f"Test: {total_num_samples_test} images")

    # with open(config_file_path, "a") as f:
    #     f.write(f"\tTEST NUM IMAGES: {total_num_samples_test}\n")

    return datasets_as_dfs


def resolve_chexpertplus_image_path(ann, root_dir, split_name):
    """Mirror Data_wrapper_chexpert.get_imagepath_from_chexpertplus."""
    path = str(ann["image_path"]).replace(".jpg", ".png")
    rel = "/".join(path.split("/")[1:])
    return os.path.join(root_dir, split_name, rel)


def get_chexpertplus_zeroshot_df(path_ann, root_dir, split_name="test"):
    """
    Build a dataframe compatible with CustomImageDataset_RegionAbnormality_test column order,
    using dummy boxes/labels (no region GT on CheXpert+). Dummy fields are unused when
    test_model_zeroshot calls object_detector(..., targets=None).
    """
    df = pd.read_excel(path_ann, engine="openpyxl")
    df = df[df["split"] == split_name].reset_index(drop=True)
    dummy_boxes = [[0, 0, 10, 10] for _ in range(29)]
    dummy_labels = list(range(1, 30))
    dummy_abn = [False] * 29

    rows = []
    missing = 0
    for _, ann in df.iterrows():
        img_path = resolve_chexpertplus_image_path(ann, root_dir, split_name)
        if not os.path.exists(img_path):
            missing += 1
            continue
        report = ann["report"] if "report" in ann and pd.notna(ann["report"]) else (
            ann["report_clean"] if "report_clean" in ann and pd.notna(ann["report_clean"]) else ""
        )
        rows.append({
            "mimic_image_file_path": img_path,
            "bbox_coordinates": dummy_boxes,
            "bbox_labels": dummy_labels,
            "bbox_is_abnormal": dummy_abn,
            "all_reference_report": report,
            "id": ann.get("id", ""),
            "image_path": ann.get("image_path", ""),
        })
    out = pd.DataFrame(rows)
    # column order required by CustomImageDataset_RegionAbnormality_test (iloc positions)
    out = out[["mimic_image_file_path", "bbox_coordinates", "bbox_labels", "bbox_is_abnormal", "all_reference_report", "id", "image_path"]]
    log.info(f"CheXpert+ {split_name}: {len(out)} images (missing files skipped: {missing})")
    return out


class CustomImageDataset_CheXpertPlus_zeroshot(torch.utils.data.Dataset):
    """Image-only dataset for zero-shot inference (no GT boxes needed)."""

    def __init__(self, dataset_df, transforms):
        self.dataset_df = dataset_df
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset_df)

    def __getitem__(self, index):
        image_path = self.dataset_df.iloc[index]["mimic_image_file_path"]
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            return None
        # transforms still expect bboxes; pass dummy (unused in zeroshot forward)
        dummy_boxes = [[0, 0, 10, 10] for _ in range(29)]
        dummy_labels = list(range(1, 30))
        transformed = self.transforms(image=image, bboxes=dummy_boxes, class_labels=dummy_labels)
        return {
            "image": transformed["image"],
            "boxes": torch.tensor(transformed["bboxes"], dtype=torch.float),
            "labels": torch.tensor(transformed["class_labels"], dtype=torch.int64),
            "bbox_is_abnormal": torch.zeros(29, dtype=torch.bool),
            "report": self.dataset_df.iloc[index]["all_reference_report"],
            "file_id": image_path,
        }


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
        # shutil.rmtree(run_folder_path)
        log.error("Delete the folder.")
        # log.error("Delete the folder or change the run number.")
        return None

    os.mkdir(run_folder_path)
    os.mkdir(weights_folder_path)
    os.mkdir(tensorboard_folder_path)

    log.info(f"Run {RUN} folder created at {run_folder_path}.")

    config_file_path = os.path.join(run_folder_path, "run_config.txt")
    config_parameters = {
        # "COMMENT": RUN_COMMENT,
        "SEED": SEED,
        "IMAGE_INPUT_SIZE": IMAGE_INPUT_SIZE,
        # "PERCENTAGE_OF_TRAIN_SET_TO_USE": PERCENTAGE_OF_TRAIN_SET_TO_USE,
        # "PERCENTAGE_OF_VAL_SET_TO_USE": PERCENTAGE_OF_VAL_SET_TO_USE,
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
        normality_count += dict_normal_abnormality[i]['normal'] # train:2808310 val:149534 test:244534
        abnormality_count += dict_normal_abnormality[i]['abnormal'] # train:437834 (6.4:1) val:23248(6.4:1) test:36766(6.6:1)
    return dict_normal_abnormality, normality_count, abnormality_count


def main():
    ## CheXpert+ zero-shot (no region GT; do not filter by `available`)
    WRITETYPE = "chexpertplus_test"
    path_ann = str(Path(__import__('os').environ.get('DATA_ROOT', './data')))
    root_dir = str(Path(__import__('os').environ.get('DATA_ROOT', './data')))
    path_to_write_root = str(Path(__import__('os').environ.get('DATA_ROOT', './data')))
    os.makedirs(path_to_write_root, exist_ok=True)

    datasets_as_dfs = get_chexpertplus_zeroshot_df(path_ann, root_dir, split_name="test")
    test_transforms = get_transforms("test")
    test_dataset = CustomImageDataset_CheXpertPlus_zeroshot(datasets_as_dfs, test_transforms)
    test_loader = get_data_loaders(test_dataset)

    path_to_best_region_abnormality_weights = str(
        Path(__import__("os").environ.get(
            "REGION_ABNORMALITY_WEIGHTS",
            str(Path(__import__("os").environ.get("DATA_ROOT", "./data")) / "weights" / "region_abnormality.pth"),
        ))
    )
    model = RegionAbnormalityClassifier()
    load_model = torch.load(path_to_best_region_abnormality_weights, map_location="cpu")
    model.load_state_dict(load_model)
    print("Loaded weights from {}".format(path_to_best_region_abnormality_weights))
    model.to(device, non_blocking=True)

    log.info("\nStarting zero-shot {} on {}!\n".format(WRITETYPE, device))
    test_model_zeroshot(
        model=model,
        test_dl=test_loader,
        writepath_root=path_to_write_root,
        data_origin=datasets_as_dfs,
        writetype=WRITETYPE,
    )


if __name__ == "__main__":

    main()
