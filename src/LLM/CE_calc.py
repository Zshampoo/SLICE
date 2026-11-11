from pathlib import Path
import ast
import csv
import hashlib
import os
import re
import tempfile
import random
import requests
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from torch import nn
from collections import OrderedDict

from src.CheXbert.src.constants import CONDITIONS
from src.CheXbert.src.label import label
from src.CheXbert.src.models.bert_labeler import bert_labeler
from src.dataset.constants import ANATOMICAL_REGIONS
from src.full_model.evaluate_full_model.evaluate_language_model import compute_NLG_scores

from deep_translator import GoogleTranslator
import difflib

device = 'cuda:0'
path_chexbert_weights = str(Path(__import__("os").environ.get("CHEXBERT_WEIGHTS", Path(__import__("os").environ.get("DATA_ROOT", "./data")) / "weights" / "chexbert.pth")))

# Keep CheXbert labeler device in sync (label.py may prefer cuda:1 on multi-GPU)
try:
    import src.CheXbert.src.label as _chexbert_label_mod
    if torch.cuda.is_available():
        _dev = torch.device("cuda:0")
        _chexbert_label_mod.device = _dev
        device = str(_dev)
    else:
        _chexbert_label_mod.device = torch.device("cpu")
        device = "cpu"
except Exception:
    pass

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def rearrange_labels(labels):
    label_cols = ['No Finding',
              'Enlarged Cardiomediastinum',
              'Cardiomegaly',
              'Lung Opacity',
              'Lung Lesion',
              'Edema',
              'Consolidation',
              'Pneumonia',
              'Atelectasis',
              'Pneumothorax',
              'Pleural Effusion',
              'Pleural Other',
              'Fracture',
              'Support Devices']
    num_to_condition = {0: 'blank/NaN', 1: 'positive', 2: 'negative', 3: 'uncertain'}
    num_to_condition_uones = {0: '[NEG]', 1: '[POS]', 2: '[NEG]', 3: '[POS]'}
    labels_new = []
    prompt_all = []
    for patient_label in labels:
        patient_label_new = []
        patient_prompt = []
        for lab in label_cols:
            for plab in patient_label:
                if plab[0] == lab:
                    patient_label_new.append(plab)
                    if plab[1] == 'blank/NaN' or plab[1] == 'negative':
                        patient_prompt.append('[NEG]')
                    else:
                        patient_prompt.append('[POS]')
        labels_new.append(patient_label_new)
        prompt_all.append(patient_prompt)
    return labels_new, prompt_all

def compute_language_model_scores(language_model_scores, gen_and_ref_sentences, gen_and_ref_reports):

    # def compute_sentence_level_scores():
    #     def remove_gen_sents_corresponding_to_empty_ref_sents(gen_sents, ref_sents):
    #         """
    #         We can't compute scores on generated sentences whose corresponding reference sentence is empty.
    #         So we need to discard them both.
    #         """
    #         filtered_gen_sents = []
    #         filtered_ref_sents = []
    #
    #         for gen_sent, ref_sent in zip(gen_sents, ref_sents):
    #             if ref_sent != "":
    #                 filtered_gen_sents.append(gen_sent)
    #                 filtered_ref_sents.append(ref_sent)
    #
    #         return filtered_gen_sents, filtered_ref_sents
    #
    #     def compute_sent_level_scores_for_subset(subset, gen_sents, ref_sents):
    #         nlg_metrics = ["meteor"]
    #         nlg_scores = compute_NLG_scores(nlg_metrics, gen_sents, ref_sents)
    #         meteor_score = nlg_scores["meteor"]
    #         language_model_scores[subset]["meteor"] = meteor_score
    #
    #     def compute_sent_level_scores_for_region(region_name, gen_sents, ref_sents):
    #         nlg_metrics = ["meteor"]
    #         nlg_scores = compute_NLG_scores(nlg_metrics, gen_sents, ref_sents)
    #         meteor_score = nlg_scores["meteor"]
    #         language_model_scores["region"][region_name]["meteor"] = meteor_score
    #
    #     def compute_sent_level_meteor_ratio_score(gen_sents, ref_sents):
    #         """
    #         We want to compute the ratio of the meteor scores for when:
    #         - a generated sentence is paired with its corresponding reference sentence of a given image (value for the numerator)
    #         vs
    #         - a generated sentence is paired with all other non-corresponding reference sentences of a given image (value for the denominator)
    #
    #         the numerator value is already computed by language_model_scores["all"]["meteor"], since this is exactly the meteor score for when the generated sentences
    #         are paried with their corresponding reference sentences. Hence only the denominator value has to be calculated separately
    #         """
    #         gen_sents_for_computing_meteor_ratio_score = []
    #         ref_sents_for_computing_meteor_ratio_score = []
    #
    #         # List[int] that can be used to get all generated and reference sentences that correspond to the same image
    #         num_generated_sentences_per_image = gen_and_ref_sentences["num_generated_sentences_per_image"]
    #
    #         curr_index = 0
    #         for num_gen_sents in num_generated_sentences_per_image:
    #             gen_sents_single_image = gen_sents[curr_index:curr_index + num_gen_sents]
    #             ref_sents_single_image = ref_sents[curr_index:curr_index + num_gen_sents]
    #             # the number of generated sentences per image is exactly the same as the number of "retrieved" reference sentences per image
    #             # (see function get_ref_sentences_for_selected_regions)
    #
    #             curr_index += num_gen_sents
    #
    #             gen_sents_single_image_filtered, ref_sents_single_image_filtered = remove_gen_sents_corresponding_to_empty_ref_sents(gen_sents_single_image, ref_sents_single_image)
    #
    #             # now that we have the generated and (non-empty) reference sentences for a single image,
    #             # we have to pair-wise match them except for the "correct" match between the corresponding gen and ref sents (see doc string of this function)
    #             for i, gen_sent in enumerate(gen_sents_single_image_filtered):
    #                 for j, ref_sent in enumerate(ref_sents_single_image_filtered):
    #                     if i == j:
    #                         continue  # skip "correct" match
    #                     else:
    #                         gen_sents_for_computing_meteor_ratio_score.append(gen_sent)
    #                         ref_sents_for_computing_meteor_ratio_score.append(ref_sent)
    #
    #         # compute the "denominator" meteor score
    #         nlg_metrics = ["meteor"]
    #         nlg_scores = compute_NLG_scores(nlg_metrics, gen_sents_for_computing_meteor_ratio_score, ref_sents_for_computing_meteor_ratio_score)
    #         denominator_meteor_score = nlg_scores["meteor"]
    #
    #         numerator_meteor_score = language_model_scores["all"]["meteor"]
    #
    #         language_model_scores["all"]["meteor_ratio"] = numerator_meteor_score / denominator_meteor_score
    #
    #     generated_sents = gen_and_ref_sentences["generated_sentences"]
    #     generated_sents_normal = gen_and_ref_sentences["generated_sentences_normal_selected_regions"]
    #     generated_sents_abnormal = gen_and_ref_sentences["generated_sentences_abnormal_selected_regions"]
    #
    #     reference_sents = gen_and_ref_sentences["reference_sentences"]
    #     reference_sents_normal = gen_and_ref_sentences["reference_sentences_normal_selected_regions"]
    #     reference_sents_abnormal = gen_and_ref_sentences["reference_sentences_abnormal_selected_regions"]
    #
    #     gen_sents_filtered, ref_sents_filtered = remove_gen_sents_corresponding_to_empty_ref_sents(generated_sents, reference_sents)
    #     gen_sents_normal_filtered, ref_sents_normal_filtered = remove_gen_sents_corresponding_to_empty_ref_sents(generated_sents_normal, reference_sents_normal)
    #     gen_sents_abnormal_filtered, ref_sents_abnormal_filtered = remove_gen_sents_corresponding_to_empty_ref_sents(generated_sents_abnormal, reference_sents_abnormal)
    #
    #     compute_sent_level_scores_for_subset("all", gen_sents_filtered, ref_sents_filtered)
    #     compute_sent_level_scores_for_subset("normal", gen_sents_normal_filtered, ref_sents_normal_filtered)
    #     compute_sent_level_scores_for_subset("abnormal", gen_sents_abnormal_filtered, ref_sents_abnormal_filtered)
    #
    #     compute_sent_level_meteor_ratio_score(generated_sents, reference_sents)
    #
    #     for region_index, region_name in enumerate(ANATOMICAL_REGIONS):
    #         region_generated_sentences = gen_and_ref_sentences[region_index]["generated_sentences"]
    #         region_reference_sentences = gen_and_ref_sentences[region_index]["reference_sentences"]
    #
    #         region_gen_sents_filtered, region_ref_sents_filtered = remove_gen_sents_corresponding_to_empty_ref_sents(region_generated_sentences, region_reference_sentences)
    #
    #         if len(region_gen_sents_filtered) != 0:
    #             compute_sent_level_scores_for_region(region_name, region_gen_sents_filtered, region_ref_sents_filtered)
    #         else:
    #             language_model_scores["region"][region_name]["meteor"] = -1

    def compute_report_level_scores():
        gen_reports = gen_and_ref_reports["generated_reports"]
        ref_reports = gen_and_ref_reports["reference_reports"]

        nlg_metrics = ["bleu", "meteor", "rouge", "cider"]
        nlg_scores = compute_NLG_scores(nlg_metrics, gen_reports, ref_reports)

        for nlg_metric_name, score in nlg_scores.items():
            language_model_scores["report"][nlg_metric_name] = score



    # language_model_scores = create_language_model_scores_dict()
    compute_report_level_scores()
    # compute_sentence_level_scores()

def write_clinical_efficacy_scores(ce_score_dict, writer, overall_steps_taken):
    """
    ce_score_dict is of the structure:

    {
        precision_micro_5: ...,
        precision_micro_all: ...,
        precision_example_all: ...,
        recall_micro_5: ...,
        recall_micro_all: ...,
        recall_example_all: ...,
        f1_micro_5: ...,
        f1_micro_all: ...,
        f1_example_all: ...,
        acc_micro_5: ...,
        acc_micro_all: ...,
        acc_example_all: ...,
        condition_1 : {
            precision: ...,
            recall: ...,
            f1: ...,
            acc: ...,
        },
        condition_2 : {
            precision: ...,
            recall: ...,
            f1: ...,
            acc: ...,
        },
        ...,
        condition_14 : {
            precision: ...,
            recall: ...,
            f1: ...,
            acc: ...,
        }
    }

    where the "..." after the 4 metrics are the corresponding scores,
    and condition_* are from the 14 conditions in src/CheXbert/src/constants.py
    """
    for k, v in ce_score_dict.items():
        if k.startswith("precision") or k.startswith("recall") or k.startswith("f1") or k.startswith("acc"):
            writer.add_scalar(f"language_model/report/CE/{k}", v, overall_steps_taken)
        else:
            # k is a condition
            condition_name = "_".join(k.lower().split())
            for metric, score in ce_score_dict[k].items():
                writer.add_scalar(f"language_model/report/CE/{condition_name}/{metric}", score, overall_steps_taken)

def write_language_model_scores(language_model_scores, writer, overall_steps_taken):
    """
    language_model_scores is a dict with keys:
        - all: for all generated sentences
        - normal: for all generated sentences corresponding to normal regions
        - abnormal: for all generated sentences corresponding to abnormal regions
        - report: for all generated reports
        - region: for generated sentences per region
    """
    for subset in language_model_scores:
        if subset == "region":
            for region_name in language_model_scores["region"]:
                for metric, score in language_model_scores["region"][region_name].items():
                    # replace white space by underscore for region name (i.e. "right upper lung" -> "right_upper_lung")
                    region_name_underscored = "_".join(region_name.split())
                    writer.add_scalar(f"language_model/region/{region_name_underscored}/{metric}", score, overall_steps_taken)
        else:
            for metric, score in language_model_scores[subset].items():
                if metric == "CE":
                    ce_score_dict = language_model_scores["report"]["CE"]
                    write_clinical_efficacy_scores(ce_score_dict)
                else:
                    writer.add_scalar(f"language_model/{subset}/{metric}", score, overall_steps_taken)

def compute_clinical_efficacy_scores(language_model_scores: dict, gen_reports: list[str], ref_reports: list[str], IDs=None):
    """
    This function computes:
        - micro average CE scores over all 14 conditions
        - micro average CE scores over 5 conditions ("Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion")
        -> this is done following Miura (https://arxiv.org/pdf/2010.10042.pdf)
        - (micro) average CE scores of each condition
        - example-based CE scores over all 14 conditions
        -> this is done following Nicolson (https://arxiv.org/pdf/2201.09405.pdf)

    To compute these scores, we first need to get the disease labels extracted by CheXbert for both the generated and reference reports.
    This is done by the (nested) function "get_chexbert_labels_for_gen_and_ref_reports". Inside this function, there is another function
    called "label" from the module src/CheXbert/src/label.py that extracts these labels requiring 2 input arguments:
        1. chexbert (nn.Module): instantiated chexbert model
        2. csv_path (str): path to the csv file with the reports. The csv file has to have 1 column titled "Report Impression"
        under which the reports can be found

    We use a temporary directory to create the csv files for the generated and reference reports.

    The function label returns preds_gen_reports and preds_ref_reports respectively, which are List[List[int]],
    with the outer list always having len=14 (for 14 conditions, specified in CheXbert/src/constants.py),
    and the inner list has len=num_reports.

    E.g. the 1st inner list could be [2, 1, 0, 3], which means the 1st report has label 2 for the 1st condition (which is 'Enlarged Cardiomediastinum'),
    the 2nd report has label 1 for the 1st condition, the 3rd report has label 0 for the 1st condition, the 4th and final report label 3 for the 1st condition.

    There are 4 possible labels:
        0: blank/NaN (i.e. no prediction could be made about a condition, because it was no mentioned in a report)
        1: positive (condition was mentioned as present in a report)
        2: negative (condition was mentioned as not present in a report)
        3: uncertain (condition was mentioned as possibly present in a report)

    To compute the micro average scores (i.e. all the scores except of the example-based scores), we follow the implementation of the paper
    by Miura et. al., who considered the negative and blank/NaN to be one whole negative class, and positive and uncertain to be one whole positive class.
    For reference, see lines 141 and 143 of Miura's implementation: https://github.com/ysmiura/ifcc/blob/master/eval_prf.py#L141,
    where label 3 is converted to label 1, and label 2 is converted to label 0.

    To compute the example-based scores, we follow Nicolson's implementation, who considered blank/NaN, negative and uncertain to be the negative class,
    and only positive to be the positive class. Meaning labels 2 and 3 are converted to label 0.
    """

    def get_chexbert():
        model = bert_labeler()
        checkpoint = torch.load(path_chexbert_weights, map_location=torch.device("cpu"), weights_only=True)
        state = checkpoint['model_state_dict']
        use_cpu = device == "cpu" or not torch.cuda.is_available()
        if use_cpu:
            cpu_state = OrderedDict()
            for k, v in state.items():
                nk = k[7:] if k.startswith("module.") else k
                cpu_state[nk] = v
            model.load_state_dict(cpu_state, strict=False)
            model.to(torch.device("cpu"))
            model.eval()
            return model
        # Single-GPU path: avoid DataParallel so CUDA_VISIBLE_DEVICES remapping is safe
        gpu_state = OrderedDict()
        for k, v in state.items():
            nk = k[7:] if k.startswith("module.") else k
            gpu_state[nk] = v
        model.load_state_dict(gpu_state, strict=False)
        model = model.to(device)
        model.eval()
        return model

    def get_chexbert_labels_for_gen_and_ref_reports():
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_gen_reports_file_path = os.path.join(temp_dir, "gen_reports.csv")
            csv_ref_reports_file_path = os.path.join(temp_dir, "ref_reports.csv")

            header = ["Report Impression"]

            with open(csv_gen_reports_file_path, "w") as fp:
                csv_writer = csv.writer(fp)
                csv_writer.writerow(header)
                csv_writer.writerows([[gen_report] for gen_report in gen_reports])

            with open(csv_ref_reports_file_path, "w") as fp:
                csv_writer = csv.writer(fp)
                csv_writer.writerow(header)
                csv_writer.writerows([[ref_report] for ref_report in ref_reports])

            # preds_*_reports are List[List[int]] with the labels extracted by CheXbert (see doc string for details)
            preds_gen_reports = label(chexbert, csv_gen_reports_file_path)
            preds_ref_reports = label(chexbert, csv_ref_reports_file_path)

        return preds_gen_reports, preds_ref_reports

    def compute_micro_average_CE_scores(preds_gen_reports, preds_ref_reports):
        def convert_labels_like_miura(preds_reports: list[list[int]]):
            """
            See doc string of update_clinical_efficacy_scores function for more details.
            Miura (https://arxiv.org/pdf/2010.10042.pdf) considers blank/NaN (label 0) and negative (label 2) to be the negative class,
            and positive (label 1) and uncertain (label 3) to be the positive class.

            Thus we convert label 2 -> label 0 and label 3 -> label 1.
            """
            def convert_label(label: int): # u-one
                if label == 2:
                    return 0
                elif label == 3:
                    return 1
                else:
                    return label

            preds_reports_converted = [[convert_label(label) for label in condition_list] for condition_list in preds_reports]

            return preds_reports_converted

        preds_gen_reports_converted = convert_labels_like_miura(preds_gen_reports)
        preds_ref_reports_converted = convert_labels_like_miura(preds_ref_reports)

        # for the CE scores, we follow Miura (https://arxiv.org/pdf/2010.10042.pdf) in micro averaging them over these 5 conditions:
        five_conditions_to_evaluate = {"Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"}
        conflict6_conditions_to_evaluate = {
            "Cardiomegaly", "Enlarged Cardiomediastinum", "Atelectasis",
            "Pleural Effusion", "Edema", "Lung Opacity",
        }

        total_preds_gen_reports_5_conditions = []
        total_preds_ref_reports_5_conditions = []
        total_preds_gen_reports_conflict6 = []
        total_preds_ref_reports_conflict6 = []

        # we also compute the micro average over all 14 conditions:
        total_preds_gen_reports_14_conditions = []
        total_preds_ref_reports_14_conditions = []

        # iterate over the 14 conditions
        for preds_gen_reports_condition, preds_ref_reports_condition, condition in zip(preds_gen_reports_converted, preds_ref_reports_converted, CONDITIONS):
            if condition in five_conditions_to_evaluate:
                total_preds_gen_reports_5_conditions.extend(preds_gen_reports_condition)
                total_preds_ref_reports_5_conditions.extend(preds_ref_reports_condition)

            if condition in conflict6_conditions_to_evaluate:
                total_preds_gen_reports_conflict6.extend(preds_gen_reports_condition)
                total_preds_ref_reports_conflict6.extend(preds_ref_reports_condition)

            total_preds_gen_reports_14_conditions.extend(preds_gen_reports_condition)
            total_preds_ref_reports_14_conditions.extend(preds_ref_reports_condition)

            # compute and save scores for the given condition
            precision, recall, f1, _ = precision_recall_fscore_support(preds_ref_reports_condition, preds_gen_reports_condition, average="binary")
            acc = accuracy_score(preds_ref_reports_condition, preds_gen_reports_condition)

            language_model_scores["report"]["CE"][condition]["precision"] = precision
            language_model_scores["report"]["CE"][condition]["recall"] = recall
            language_model_scores["report"]["CE"][condition]["f1"] = f1
            language_model_scores["report"]["CE"][condition]["acc"] = acc

        # compute and save scores for all 14 conditions
        precision, recall, f1, _ = precision_recall_fscore_support(total_preds_ref_reports_14_conditions, total_preds_gen_reports_14_conditions, average="binary")
        acc = accuracy_score(total_preds_ref_reports_14_conditions, total_preds_gen_reports_14_conditions)

        language_model_scores["report"]["CE"]["precision_micro_all"] = precision
        language_model_scores["report"]["CE"]["recall_micro_all"] = recall
        language_model_scores["report"]["CE"]["f1_micro_all"] = f1
        language_model_scores["report"]["CE"]["acc_all"] = acc

        # compute and save scores for the 5 conditions
        precision, recall, f1, _ = precision_recall_fscore_support(total_preds_ref_reports_5_conditions, total_preds_gen_reports_5_conditions, average="binary")
        acc = accuracy_score(total_preds_ref_reports_5_conditions, total_preds_gen_reports_5_conditions)

        language_model_scores["report"]["CE"]["precision_micro_5"] = precision
        language_model_scores["report"]["CE"]["recall_micro_5"] = recall
        language_model_scores["report"]["CE"]["f1_micro_5"] = f1
        language_model_scores["report"]["CE"]["acc_5"] = acc

        # compute and save scores for the 6 conflict/table conditions
        precision, recall, f1, _ = precision_recall_fscore_support(
            total_preds_ref_reports_conflict6, total_preds_gen_reports_conflict6, average="binary"
        )
        acc = accuracy_score(total_preds_ref_reports_conflict6, total_preds_gen_reports_conflict6)
        language_model_scores["report"]["CE"]["precision_micro_conflict_6"] = precision
        language_model_scores["report"]["CE"]["recall_micro_conflict_6"] = recall
        language_model_scores["report"]["CE"]["f1_micro_conflict_6"] = f1
        language_model_scores["report"]["CE"]["acc_micro_conflict_6"] = acc

    def compute_example_based_CE_scores(preds_gen_reports, preds_ref_reports):
        """
        example-based means precision/recall/F1/acc are computed for each report, and then these scores are averaged over all reports
        """
        preds_gen_reports_np = np.array(preds_gen_reports)  # array of shape (14 x num_reports), 14 for 14 conditions
        preds_ref_reports_np = np.array(preds_ref_reports)  # array of shape (14 x num_reports)

        # convert label 1 to True and everything else (i.e. labels 0, 2, 3) to False
        # (effectively doing the label conversion as done by Nicolson, see doc string of compute_clinical_efficacy_scores for more details)
        preds_gen_reports_np = preds_gen_reports_np == 1
        preds_ref_reports_np = preds_ref_reports_np == 1

        tp = np.logical_and(preds_gen_reports_np, preds_ref_reports_np)  # bool array of shape (14 x num_reports)
        fp = np.logical_and(preds_gen_reports_np, ~preds_ref_reports_np)  # bool array of shape (14 x num_reports)
        fn = np.logical_and(~preds_gen_reports_np, preds_ref_reports_np)  # bool array of shape (14 x num_reports)
        tn = np.logical_and(~preds_gen_reports_np, ~preds_ref_reports_np)  # bool array of shape (14 x num_reports)

        # sum up the TP, FP, FN and TN for each report (i.e. for each column)
        tp_example = tp.sum(axis=0)  # int array of shape (num_reports)
        fp_example = fp.sum(axis=0)  # int array of shape (num_reports)
        fn_example = fn.sum(axis=0)  # int array of shape (num_reports)
        tn_example = tn.sum(axis=0)  # int array of shape (num_reports)

        # compute the scores for each report
        precision_example = tp_example / (tp_example + fp_example)  # float array of shape (num_reports)
        recall_example = tp_example / (tp_example + fn_example)  # float array of shape (num_reports)
        f1_example = (2 * tp_example) / (2 * tp_example + fp_example + fn_example)  # float array of shape (num_reports)
        acc_example = (tp_example + tn_example) / (tp_example + tn_example + fp_example + fn_example)  # float array of shape (num_reports)

        # since there can be cases of zero division, we have to replace the resulting nan values with 0.0
        precision_example[np.isnan(precision_example)] = 0.0
        recall_example[np.isnan(recall_example)] = 0.0
        f1_example[np.isnan(f1_example)] = 0.0
        acc_example[np.isnan(acc_example)] = 0.0

        # finally, take the mean over the scores for all reports
        precision_example = float(precision_example.mean())
        recall_example = float(recall_example.mean())
        f1_example = float(f1_example.mean())
        acc_example = float(acc_example.mean())

        language_model_scores["report"]["CE"]["precision_example_all"] = precision_example
        language_model_scores["report"]["CE"]["recall_example_all"] = recall_example
        language_model_scores["report"]["CE"]["f1_example_all"] = f1_example
        language_model_scores["report"]["CE"]["acc_example_all"] = acc_example
    
    def pred_to_label(pred):
        num_to_condition = {0: 'blank/NaN', 1: 'positive', 2: 'negative', 3: 'uncertain'}
        Label_chexBert = ["Enlarged Cardiomediastinum",
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
                          "Support Devices",
                          "No Finding"]
        # pred_labels = []
        # for index, d in enumerate(pred):
        #     pred_labels.append([Label_chexBert[index], num_to_condition[d[0]]])
        # return pred_labels
        pred_labels = []
        for sample_index in range(len(pred[0])):
            pred_label_sample = []
            for label_index in range(len(pred)):
                pred_label_sample.append([Label_chexBert[label_index], num_to_condition[pred[label_index][sample_index]]])
            pred_labels.append(pred_label_sample)
        return pred_labels
        
    chexbert = get_chexbert()
    preds_gen_reports, preds_ref_reports = get_chexbert_labels_for_gen_and_ref_reports()

    compute_micro_average_CE_scores(preds_gen_reports, preds_ref_reports)
    compute_example_based_CE_scores(preds_gen_reports, preds_ref_reports)

    pred_gen_labels = pred_to_label(preds_gen_reports)
    pred_ref_labels = pred_to_label(preds_ref_reports)
    pred_gen_labels_values = [[y[1] for y in x] for x in pred_gen_labels]
    pred_ref_labels_values = [[y[1] for y in x] for x in pred_ref_labels]
    # pred_labels_new, pred_prompt_all = rearrange_labels(pred_gen_labels)
    # ref_labels_new, ref_prompt_all = rearrange_labels(pred_ref_labels)
    # if IDs is not None:
    #     df = pd.DataFrame({
    #         'path': IDs,
    #         'gen_labels': pred_labels_new,
    #         'ref_labels': ref_labels_new,
    #         'gen_prompt': pred_prompt_all
    #     })
    #     df.to_excel(os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))), index=False)
    return pred_gen_labels, pred_ref_labels

def create_language_model_scores_dict():
    language_model_scores = {}

    # on report-level, we evalute on:
    # BLEU 1-4
    # METEOR
    # ROUGE-L
    # Cider-D
    # CE scores (P, R, F1, acc)
    language_model_scores["report"] = {f"bleu_{i}": None for i in range(1, 5)}
    language_model_scores["report"]["meteor"] = None
    language_model_scores["report"]["rouge"] = None
    language_model_scores["report"]["cider"] = None
    language_model_scores["report"]["CE"] = {
        # following Miura (https://arxiv.org/pdf/2010.10042.pdf), we evaluate the micro average CE scores over these 5 diseases/conditions:
        # "Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"
        "precision_micro_5": None,
        "recall_micro_5": None,
        "f1_micro_5": None,
        "acc_5": None,

        # we additionally compute the micro average CE scores over all conditions
        "precision_micro_all": None,
        "recall_micro_all": None,
        "f1_micro_all": None,
        "acc_all": None,

        # micro average over 6 conflict/table diseases
        "precision_micro_conflict_6": None,
        "recall_micro_conflict_6": None,
        "f1_micro_conflict_6": None,
        "acc_micro_conflict_6": None,
    }

    # we also compute the CE scores for each of the 14 conditions individually
    for condition in CONDITIONS:
        language_model_scores["report"]["CE"][condition] = {
            "precision": None,
            "recall": None,
            "f1": None,
            "acc": None
        }

    # following Nicolson (https://arxiv.org/pdf/2201.09405.pdf), we evaluate the example-based CE scores over all conditions
    # example-based means precision/recall/F1/acc are computed for each report, and then these scores are averaged over all reports
    language_model_scores["report"]["CE"]["precision_example_all"] = None
    language_model_scores["report"]["CE"]["recall_example_all"] = None
    language_model_scores["report"]["CE"]["f1_example_all"] = None
    language_model_scores["report"]["CE"]["acc_example_all"] = None

    # on sentence-level, we only evaluate on METEOR, since this metric gives meaningful scores on sentence-level (as opposed to e.g. BLEU)
    # we distinguish between generated sentences for all, normal, and abnormal regions
    for subset in ["all", "normal", "abnormal"]:
        language_model_scores[subset] = {"meteor": None}

    # we also compute these scores for each region individually
    language_model_scores["region"] = {}
    for region_name in ANATOMICAL_REGIONS:
        language_model_scores["region"][region_name] = {"meteor": None}

    # and finally, on sentence-level we also compute the ratio of the meteor scores for when:
    #   - a generated sentence is paired with its corresponding reference sentence of a given image (value for the numerator)
    #   vs
    #   - a generated sentence is paired with all other non-corresponding reference sentences of a given image (value for the denominator)
    #
    # the numerator value is already computed by language_model_scores["all"]["meteor"], since this is exactly the meteor score for when the generated sentences
    # are paired with their corresponding reference sentences. Hence only the denominator value has to be calculated separately
    language_model_scores["all"]["meteor_ratio"] = None

    return language_model_scores

def compute_report_level_scores(language_model_scores, gen_reports, ref_reports):

    nlg_metrics = ["bleu", "meteor", "rouge", "cider"]
    # nlg_metrics = ["bleu",  "rouge", "cider"]
    nlg_scores = compute_NLG_scores(nlg_metrics, gen_reports, ref_reports)

    for nlg_metric_name, score in nlg_scores.items():
        language_model_scores["report"][nlg_metric_name] = score

def write_scores(path, language_model_scores):
    with open(path, 'w', encoding="utf-8") as f:
        # NLG
        flag_micro_5, flag_micro_all, flag_example_all, flag_conflict6 = False, False, False, False
        for metric, score in language_model_scores['report'].items():
            if metric == "CE":
                ce_score_dict = language_model_scores["report"]["CE"]
                for k, v in ce_score_dict.items():
                    if k.startswith("precision_micro_5") or k.startswith("recall_micro_5") or k.startswith("f1_micro_5") or k.startswith("acc_5"):
                        if not flag_micro_5:
                            f.write(f"\nCE MICRO 5\n")
                            flag_micro_5 = True
                        f.write(f"CE/{k}:{v}\n")
                    elif k.startswith("precision_micro_all") or k.startswith("recall_micro_all") or k.startswith("f1_micro_all") or k.startswith("acc_all"):
                        if not flag_micro_all:
                            f.write(f"\nCE MICRO ALL\n")
                            flag_micro_all = True
                        f.write(f"CE/{k}:{v}\n")
                    elif (
                        k.startswith("precision_micro_conflict_6")
                        or k.startswith("recall_micro_conflict_6")
                        or k.startswith("f1_micro_conflict_6")
                        or k.startswith("acc_micro_conflict_6")
                    ):
                        if not flag_conflict6:
                            f.write(f"\nCE MICRO CONFLICT 6\n")
                            flag_conflict6 = True
                        f.write(f"CE/{k}:{v}\n")
                    elif k.startswith("precision_example_all") or k.startswith("recall_example_all") or k.startswith("f1_example_all") or k.startswith("acc_example_all"):
                        if not flag_example_all:
                            f.write(f"\nCE EXAMPLE ALL\n")
                            flag_example_all = True
                        f.write(f"CE/{k}:{v}\n")
                    elif isinstance(v, dict):
                        # k is a condition
                        condition_name = "_".join(k.lower().split())
                        for metric, score in v.items():
                            f.write(f"CE/{condition_name}/{metric}:{score}\n")
                    else:
                        # scalar CE metric we don't specially section
                        f.write(f"CE/{k}:{v}\n")
            else:
                f.write(f"language_model/{metric}:{score}\n")
        # 调序

# def translate_en_to_zh(text):
#     translated = GoogleTranslator(source='en', target='zh-CN').translate(text)
#     # print("翻译：", translated)
#     return translated


def translate_en_to_zh(text, max_retries=3, delay=2, print_result=False):
    """
    使用 GoogleTranslator 进行英文到中文的翻译，支持失败重试。

    参数：
        text (str): 要翻译的英文文本
        max_retries (int): 最大重试次数
        delay (int): 每次重试之间的秒数
        print_result (bool): 是否打印翻译结果
    返回：
        str: 翻译后的中文，或 "[翻译失败]"
    """
    for attempt in range(1, max_retries + 1):
        try:
            translated = GoogleTranslator(source='en', target='zh-CN').translate(text)
            if print_result:
                print(f"翻译成功：{translated}")
            return translated
        except Exception as e:
            print(f"翻译失败，第 {attempt} 次尝试，错误信息：{e}")
            if attempt < max_retries:
                time.sleep(delay)
            else:
                return "[翻译失败]"

def translate_en_to_zh_baidu(text, appid = '20250511002354644', secret_key = 'Tuyf1drc58j6lcnaIO7S', print_result=True, retry=3, sleep=1):
    """
    使用百度翻译 API，将英文翻译为中文。
    """
    url = "https://fanyi-api.baidu.com/api/trans/vip/translate"
    salt = str(random.randint(10000, 99999))
    sign = hashlib.md5((appid + text + salt + secret_key).encode('utf-8')).hexdigest()

    params = {
        'q': text,
        'from': 'en',
        'to': 'zh',
        'appid': appid,
        'salt': salt,
        'sign': sign
    }

    for attempt in range(retry):
        try:
            response = requests.get(url, params=params, timeout=5)
            result = response.json()
            if 'trans_result' in result:
                translated = result['trans_result'][0]['dst']
                if print_result:
                    print(f"翻译：{translated}")
                return translated
            else:
                print(f"翻译失败：{result}")
                return "[翻译失败]"
        except Exception as e:
            print(f"网络错误，第 {attempt+1} 次尝试失败：{e}")
            time.sleep(sleep)
    return "[网络错误]"


def split_sentences(text):
    # 按句号分割并清理
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    return sentences

def fuzzy_match(s1_list, s2_list, threshold=0.85):
    matched_s1 = set()
    matched_s2 = set()
    common = []

    for i, s1 in enumerate(s1_list):
        for j, s2 in enumerate(s2_list):
            if j in matched_s2:
                continue
            ratio = difflib.SequenceMatcher(None, s1, s2).ratio()
            if ratio >= threshold:
                common.append(s1 if len(s1) < len(s2) else s2)  # 取较短那个
                matched_s1.add(i)
                matched_s2.add(j)
                break

    return matched_s1, matched_s2, common

def process_strings(text1, text2, threshold=0.85):
    sents1 = split_sentences(text1)
    sents2 = split_sentences(text2)

    matched1_idx, matched2_idx, common = fuzzy_match(sents1, sents2, threshold)

    # 保留未匹配的句子并拼接成新字符串
    processed1 = '. '.join([sents1[i] for i in range(len(sents1)) if i not in matched1_idx])
    processed2 = '. '.join([sents2[i] for i in range(len(sents2)) if i not in matched2_idx])
    common_str = '. '.join(common)

    # 如果原始字符串有句号结尾，保持它
    if text1.strip().endswith('.'):
        processed1 += '.'
    if text2.strip().endswith('.'):
        processed2 += '.'
    if common and (text1.strip().endswith('.') or text2.strip().endswith('.')):
        common_str += '.'

    return processed1, processed2, common_str

def ensure_list(x):
    return x if isinstance(x, list) else [x]

def calc_samplelevel_metric(report_llm, report_r2g, report_gt):
    language_model_scores_llm, language_model_scores_r2g = create_language_model_scores_dict(), create_language_model_scores_dict()
    report_llm, report_r2g, report_gt = ensure_list(report_llm), ensure_list(report_r2g), ensure_list(report_gt)

    compute_report_level_scores(language_model_scores_llm, report_llm, report_gt)
    compute_report_level_scores(language_model_scores_r2g, report_r2g, report_gt)

    preds_llm_gen,  preds_llm_ref = compute_clinical_efficacy_scores(language_model_scores_llm, report_llm, report_gt)
    preds_r2g_gen,  preds_r2g_ref = compute_clinical_efficacy_scores(language_model_scores_r2g, report_r2g, report_gt)

    assert preds_llm_ref == preds_r2g_ref, f"Lists are not equal: \nllm {preds_llm_ref}\n != \nr2g {preds_r2g_ref}\n"
    
    return language_model_scores_llm, language_model_scores_r2g, preds_llm_gen, preds_r2g_gen, preds_llm_ref

def color_substrings(text, substrings, color="red"):
    for sub in substrings:
        if sub in text:
            text = text.replace(
                sub,
                f'<span style="color:{color}">{sub}</span>'
            )
    return text

# def write_to_txt():

def analysis_report(readpath, writepath, baseline_name, col_gt, col_bas, col_pred_reginfo, col_pred_disinfo, col_label_reginfo, col_label_disinfo,
                    col_bas_llm_pred, col_bas_llm_label, col_bas_trans_pred, col_bas_trans_label, ids = None):
    # df = pd.read_csv(readpath)
    df = pd.read_excel(readpath, engine="openpyxl")
    # df = df[82:]
    # df = df[:20]
    THRESHOLD = 0.8
    # df['label_region'] = df['label_region'].apply(ast.literal_eval)
    # df['label_disease'] = df['label_disease'].apply(ast.literal_eval)
    # df['pred_region_info_15w'] = df['pred_region_info_15w'].apply(ast.literal_eval)
    # df['pred_disease_info_chexber_uone_15w'] = df['pred_disease_info_chexber_uone_15w'].apply(ast.literal_eval)

    df[col_label_reginfo] = df[col_label_reginfo].apply(ast.literal_eval)
    df[col_label_disinfo] = df[col_label_disinfo].apply(ast.literal_eval)
    df[col_pred_reginfo] = df[col_pred_reginfo].apply(ast.literal_eval)
    df[col_pred_disinfo] = df[col_pred_disinfo].apply(ast.literal_eval)

    ztg_list, ztr_list, ztl_list, zptl_list, zptr_list, zcommonstr_list, labelr_list, labeld_list = [], [], [], [], [], [], [], []
    score_baseline_bleu1_list, score_baseline_bleu4_list, score_baseline_rouge_list, score_baseline_CEf1_list = [], [], [], []
    score_llm_bleu1_list, score_llm_bleu4_list, score_llm_rouge_list, score_llm_CEf1_list = [], [], [], []
    label_llm_list, label_baseline_list, label_ref_list = [], [], []
    llm_is_better_NLG_bleu_list, llm_is_better_CE_f1_list = [], []
    # gpt_output_label baselineen_report gt_report
    condition_dict = {x: 0 for x in range(16)}

    savepath = f'{writepath}/TXT'
    os.makedirs(savepath, exist_ok=True)
    for index, row in df.iterrows(): # index 85
        path = row['Path']
        id = path.split('/')[-1].split('.')[0]
        if id not in ids:
            continue
        # labels
        label_region = row[col_label_reginfo]
        label_disease = row[col_label_disinfo]
        label_region_p = [x for x in label_region if 'Abnormal' in x]
        label_disease_p = [x for x in label_disease if 'Positive' in x]

        # preds
        pred_region = row[col_pred_reginfo]
        pred_disease = row[col_pred_disinfo]
        pred_region_p = [x for x in pred_region if 'Abnormal' in x]
        pred_disease_p = [x for x in pred_disease if 'Positive' in x]


        # tp, tpt, tr, tg = row['gpt_output_pred_15w_clean'], row['transfer_output_pred_15w'], row['baselineen_report'], row['gt_report']
        # tl, tlt = row['gpt_output_label_clean'], row['transfer_output_label_clean']

        tp, tpt, tr, tg = row[col_bas_llm_pred], row[col_bas_trans_pred], row[col_bas], row[col_gt]
        tl, tlt = row[col_bas_llm_label], row[col_bas_trans_label]

        ztp = translate_en_to_zh(tp)
        ztpt = translate_en_to_zh(tpt)
        ztr = translate_en_to_zh(tr)
        ztg = translate_en_to_zh(tg)
        ztl = translate_en_to_zh(tl)
        ztlt = translate_en_to_zh(tlt)

        ## common strs and left strs
        ## ptl, ptr, common_str = process_strings(tl, tr, THRESHOLD)
        ## zcommon_str = translate_en_to_zh(common_str)
        ## zptl = translate_en_to_zh(ptl)
        ## zptr = translate_en_to_zh(ptr)

        score_llm_wpred, score_baseline, label_llm_wpred, label_baseline, label_ref = calc_samplelevel_metric(tp, tr, tg)
        score_llm_wpred_trans, _, label_llm_wpred_trans, _, _ = calc_samplelevel_metric(tpt, tp, tg)
        score_llm_wlabel, _, label_llm_wlabel, _, _ = calc_samplelevel_metric(tl, tp, tg)
        score_llm_wlabel_trans, _, label_llm_wlabel_trans, _, _ = calc_samplelevel_metric(tlt, tp, tg)
        '''
        0: NLG上 pred比baseline好, trans比baseline好 CE上 pred比baseline好, trans比baseline好
        1: NLG上 pred比baseline差, trans比baseline好 CE上 pred比baseline好, trans比baseline好
        2: NLG上 pred比baseline好, trans比baseline差 CE上 pred比baseline好, trans比baseline好
        3: NLG上 pred比baseline好, trans比baseline好 CE上 pred比baseline差, trans比baseline好
        4: NLG上 pred比baseline好, trans比baseline好 CE上 pred比baseline好, trans比baseline差
        '''

        bleu1_p, bleu1_baseline, bleu1_t = score_llm_wpred['report']['bleu_1'], score_baseline['report']['bleu_1'], score_llm_wpred_trans['report']['bleu_1']
        rouge_p, rouge_baseline, rouge_t = score_llm_wpred['report']['rouge'], score_baseline['report']['rouge'], score_llm_wpred_trans['report']['rouge']
        meteor_p, meteor_baseline, meteor_t = score_llm_wpred['report']['meteor'], score_baseline['report']['meteor'], score_llm_wpred_trans['report']['meteor']
        cef1_p, cef1_baseline, cef1_t = score_llm_wpred['report']['CE']['f1_micro_all'], score_baseline['report']['CE']['f1_micro_all'], score_llm_wpred_trans['report']['CE']['f1_micro_all']

        llm_pred_is_better_than_baseline_NLG_bleu = (bleu1_baseline < bleu1_p) & (rouge_baseline < rouge_p) & (meteor_baseline < meteor_p)
        # llm_pred_is_better_than_baseline_CE_f1 = cef1_baseline <= cef1_p

        llm_trans_is_better_than_baseline_NLG_bleu = (bleu1_baseline < bleu1_t) & (rouge_baseline < rouge_t) & (meteor_baseline < meteor_t)
        llm_trans_is_better_than_baseline_CE_f1 = cef1_baseline <= cef1_t

        llm_trans_is_better_than_pred_NLG_bleu = (bleu1_p < bleu1_t) & (rouge_p < rouge_t) & (meteor_p < meteor_t)
        # llm_trans_is_better_than_pred_CE_f1 = cef1_p <= cef1_t

        # if bleu1_p >= bleu1_baseline:
        #     if bleu1_t >= bleu1_baseline:
        #         if cef1_p >= cef1_baseline:
        #             if cef1_t >= cef1_baseline:
        #                 condition_dict[0]+=1
        #             else:
        #                 condition_dict[1]+=1
        #         else:
        #             if cef1_t >= cef1_baseline:
        #                 condition_dict[2]+=1
        #             else:
        #                 condition_dict[3]+=1
        #     else:
        #         if cef1_p >= cef1_baseline:
        #             if cef1_t >= cef1_baseline:
        #                 condition_dict[4]+=1 # bleu1_p >= bleu1_baseline bleu1_t < bleu1_baseline cef1_p >= cef1_baseline cef1_t >= cef1_baseline
        #             else:
        #                 condition_dict[5]+=1
        #         else:
        #             if cef1_t >= cef1_baseline:
        #                 condition_dict[6]+=1
        #             else:
        #                 condition_dict[7]+=1
        # else:
        #     if bleu1_t >= bleu1_baseline:
        #         if cef1_p >= cef1_baseline:
        #             if cef1_t >= cef1_baseline:
        #                 condition_dict[8]+=1
        #             else:
        #                 condition_dict[9]+=1
        #         else:
        #             if cef1_t >= cef1_baseline:
        #                 condition_dict[10]+=1
        #             else:
        #                 condition_dict[11]+=1
        #     else:
        #         if cef1_p >= cef1_baseline:
        #             if cef1_t >= cef1_baseline:
        #                 condition_dict[12]+=1 # bleu1_p < bleu1_baseline bleu1_t < bleu1_baseline cef1_p >= cef1_baseline cef1_t >= cef1_baseline
        #             else:
        #                 condition_dict[13]+=1
        #         else:
        #             if cef1_t >= cef1_baseline:
        #                 condition_dict[14]+=1
        #             else:
        #                 condition_dict[15]+=1



        # llm_is_better_NLG_bleu_list.append(llm_pred_is_better_NLG_bleu)
        # llm_is_better_CE_f1_list.append(llm_pred_is_better_CE_f1)

        # score_baseline_bleu1_list.append(score_baseline['report']['bleu_1'])
        # score_baseline_bleu4_list.append(score_baseline['report']['bleu_4'])
        # score_baseline_rouge_list.append(score_baseline['report']['rouge'])
        # score_baseline_CEf1_list.append(score_baseline['report']['CE']['f1_micro_all'])
        
        # score_llm_bleu1_list.append(score_llm['report']['bleu_1'])
        # score_llm_bleu4_list.append(score_llm['report']['bleu_4'])
        # score_llm_rouge_list.append(score_llm['report']['rouge'])
        # score_llm_CEf1_list.append(score_llm['report']['CE']['f1_micro_all'])
        #
        # label_llm_list.append(preds_label_llm)
        # label_baseline_list.append(preds_label_baseline)
        # label_ref_list.append(preds_label_ref)

        # ztl_list.append([tl, ztl])
        # ztr_list.append([tr, ztr])
        # ztg_list.append([tg, ztg])

        # zptl_list.append([ptl, zptl])
        # zptr_list.append([ptr, zptr])
        # zcommonstr_list.append([common_str, zcommon_str])

        # labelr_list.append(label_region_p)
        # labeld_list.append(label_disease_p)

        # write
        CONDITION = ''
        if (llm_pred_is_better_than_baseline_NLG_bleu and llm_trans_is_better_than_pred_NLG_bleu and llm_trans_is_better_than_baseline_CE_f1):
            CONDITION = 'LLM_trans_isbetter_NLG_CE'
        elif not (llm_pred_is_better_than_baseline_NLG_bleu or llm_trans_is_better_than_baseline_NLG_bleu):
            CONDITION = f'{baseline_name}_isbetter_NLG'
        with open(f'{savepath}/{id}_{CONDITION}.txt', 'w', encoding='utf-8') as f:
            str = f'ID: {id}\n'
            # str += f'llm_is_better_NLG_bleu: {llm_is_better_NLG_bleu}\n'
            # str += f'llm_is_better_CE_f1: {llm_is_better_CE_f1}\n'
            str += f'GT report: {tg}\n\t{ztg}\n\tLabel (by chexbert):{label_ref}\n'

            bleu1_baseline, bleu4_baseline, rouge_baseline, meteor_baseline, cef1_baseline = score_baseline['report']['bleu_1'], score_baseline['report']['bleu_4'], score_baseline['report']['rouge'], score_baseline['report']['meteor'], score_baseline['report']['CE']['f1_micro_all']
            str += f'{baseline_name} report: {tr}\n\t{ztr}\n\tBLEU_1:{bleu1_baseline}  BLEU_4:{bleu4_baseline}  ROUGE_L:{rouge_baseline} METEOR:{meteor_baseline}\n\tLabel (by chexbert):{label_baseline}\n\tCE_f1:{cef1_baseline}\n'

            bleu1_llm_p, bleu4_llm_p, rouge_llm_p, meteor_llm_p, cef1_llm_p = score_llm_wpred['report']['bleu_1'], score_llm_wpred['report']['bleu_4'], score_llm_wpred['report']['rouge'], score_llm_wpred['report']['meteor'], score_llm_wpred['report']['CE']['f1_micro_all']
            str += f'LLM_pred report: {tp}\n\t{ztp}\n\tBLEU_1:{bleu1_llm_p}  BLEU_4:{bleu4_llm_p}  ROUGE_L:{rouge_llm_p} METEOR:{meteor_llm_p}\n\tLabel (by chexbert):{label_llm_wpred}\n\tCE_f1:{cef1_llm_p}'

            bleu1_llm_pt, bleu4_llm_pt, rouge_llm_pt, meteor_llm_pt, cef1_llm_pt = score_llm_wpred_trans['report']['bleu_1'], score_llm_wpred_trans['report']['bleu_4'], score_llm_wpred_trans['report']['rouge'], score_llm_wpred_trans['report']['meteor'], score_llm_wpred_trans['report']['CE']['f1_micro_all']
            str += f'\nLLM_pred_trans report: {tpt}\n\t{ztpt}\n\tBLEU_1:{bleu1_llm_pt}  BLEU_4:{bleu4_llm_pt}  ROUGE_L:{rouge_llm_pt} METEOR:{meteor_llm_pt}\n\tLabel (by chexbert):{label_llm_wpred_trans}\n\tCE_f1:{cef1_llm_pt}'

            bleu1_llm_l, bleu4_llm_l, rouge_llm_l, meteor_llm_l, cef1_llm_l = score_llm_wlabel['report']['bleu_1'], score_llm_wlabel['report']['bleu_4'], score_llm_wlabel['report']['rouge'], score_llm_wlabel['report']['meteor'], score_llm_wlabel['report']['CE']['f1_micro_all']
            str += f'\nLLM_label report: {tl}\n\t{ztl}\n\tBLEU_1:{bleu1_llm_l}  BLEU_4:{bleu4_llm_l}  ROUGE_L:{rouge_llm_l} METEOR:{meteor_llm_l}\n\tLabel (by chexbert):{label_llm_wlabel}\n\tCE_f1:{cef1_llm_l}'

            bleu1_llm_lt, bleu4_llm_lt, rouge_llm_lt, meteor_llm_lt, cef1_llm_lt = score_llm_wlabel_trans['report']['bleu_1'], score_llm_wlabel_trans['report']['bleu_4'], score_llm_wlabel_trans['report']['rouge'], score_llm_wlabel_trans['report']['meteor'], score_llm_wlabel_trans['report']['CE']['f1_micro_all']
            str += f'\nLLM_label_trans report: {tlt}\n\t{ztlt}\n\tBLEU_1:{bleu1_llm_lt}  BLEU_4:{bleu4_llm_lt}  ROUGE_L:{rouge_llm_lt} METEOR:{meteor_llm_lt}\n\tLabel (by chexbert):{label_llm_wlabel_trans}\n\tCE_f1:{cef1_llm_lt}'
            # str += f'\nCOMMON_STRING (THRESHOLD:{THRESHOLD}): {common_str}\n\t{zcommon_str}\n'
            # str += f'\nbaselineen left report : {ptr}\n\t{zptr}\n'
            # str += f'\nLLM left report : {ptl}\n\t{zptl}\n'
            str += f'\nlabel_region_abnormal: {label_region_p}\n'
            str += f'\nlabel_disease_positive: {label_disease_p}\n'
            str += f'\npred_region_abnormal: {pred_region_p}\n'
            str += f'\npred_disease_positive: {pred_disease_p}\n'
            f.writelines(str)

    df.insert(0, 'llm_is_better_NLG_bleu', llm_is_better_NLG_bleu_list)
    df.insert(0, 'llm_is_better_CE_f1', llm_is_better_CE_f1_list)
    df.insert(0, 'label_abnormal_regions', labelr_list)
    df.insert(0, 'label_positive_diseases', labeld_list)

    df.insert(0, f'score_{baseline_name}_bleu1', score_baseline_bleu1_list)
    df.insert(0, f'score_{baseline_name}_bleu4', score_baseline_bleu4_list)
    df.insert(0, f'score_{baseline_name}_rouge', score_baseline_rouge_list)
    df.insert(0, f'score_{baseline_name}_CE_f1', score_baseline_CEf1_list)
    df.insert(0, f'label_{baseline_name}_by_chexbert', label_baseline_list)
    df.insert(0, f'report_{baseline_name}_wzh', ztr_list)

    df.insert(0, 'score_llm_bleu1', score_llm_bleu1_list)
    df.insert(0, 'score_llm_bleu4', score_llm_bleu4_list)
    df.insert(0, 'score_llm_rouge', score_llm_rouge_list)
    df.insert(0, 'score_llm_CE_f1', score_llm_CEf1_list)
    df.insert(0, 'label_llm_by_chexbert', label_llm_list)
    df.insert(0, 'report_llm_wzh', ztl_list)

    df.insert(0, 'label_gt_by_chexbert', label_ref_list)
    df.insert(0, 'report_gt_wzh', ztg_list)

    df.insert(0, 'report_llmonly_wzh', zptl_list)
    df.insert(0, f'report_{baseline_name}only_wzh', zptr_list)
    df.insert(0, f'report_commonstr_{baseline_name}_llm_wzh_{THRESHOLD}', zcommonstr_list)

    df.to_csv(writepath)
    return None

def clean_report_mimic_cxr(report):
    report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
        .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
        .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
        .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
        .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
        .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
        .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
        .strip().lower().split('. ')
    sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                    .replace('\\', '').replace("'", '').strip().lower())
    tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
    report = ' . '.join(tokens) + ' .'
    return report


def calc_NLG_CE_metrics(path_to_read, path_to_write, use_transfer = True, use_clean=True):
    df = pd.read_excel(path_to_read, engine='openpyxl')
    language_model_scores_r2g = create_language_model_scores_dict()
    language_model_scores_pred = create_language_model_scores_dict()
    language_model_scores_label = create_language_model_scores_dict()
    language_model_scores_trans_pred = create_language_model_scores_dict()
    language_model_scores_trans_label = create_language_model_scores_dict()
    if use_clean:
        r2g_reports, llm_reports_pred, ref_reports, llm_reports_label = df['r2gen_report_clean'].tolist(), \
                                                     df['gpt_output_pred_15w_clean'].tolist(), \
                                                     df['gt_report_clean'].tolist(),\
                                                     df['gpt_output_label_clean'].tolist(),

        if use_transfer:
            transfer_reports_pred, transfer_reports_label = df['transfer_output_pred_15w_clean'].tolist(),  df['transfer_output_label_clean'].tolist()
    else:
        r2g_reports, llm_reports_pred, ref_reports, llm_reports_label = df['r2gen_report'].tolist(), \
                                                     df['gpt_output_pred_15w'].tolist(), \
                                                     df['gt_report'].tolist(), \
                                                     df['gpt_output_label'].tolist()
        if use_transfer:
            transfer_reports_pred, transfer_reports_label  = df['transfer_output_pred_15w'].tolist(),  df['transfer_output_label'].tolist()

    # # # r2g
    # compute_clinical_efficacy_scores(language_model_scores_r2g, r2g_reports, ref_reports)
    # compute_report_level_scores(language_model_scores_r2g, r2g_reports, ref_reports)
    # write_scores(path_to_write+'test_r2g_result_0513.txt', language_model_scores_r2g)
    # # pred
    # compute_clinical_efficacy_scores(language_model_scores_pred, llm_reports_pred, ref_reports)
    # compute_report_level_scores(language_model_scores_pred, llm_reports_pred, ref_reports)
    # write_scores(path_to_write + 'test_llm_result_pred_0513.txt', language_model_scores_pred)
    # # # label
    # compute_clinical_efficacy_scores(language_model_scores_label, llm_reports_label, ref_reports)
    # compute_report_level_scores(language_model_scores_label, llm_reports_label, ref_reports)
    # write_scores(path_to_write + 'test_llm_result_label_0513.txt', language_model_scores_label)
    # # transfer pred
    # compute_clinical_efficacy_scores(language_model_scores_trans_pred, transfer_reports_pred, ref_reports)
    # compute_report_level_scores(language_model_scores_trans_pred, transfer_reports_pred, ref_reports)
    # write_scores(path_to_write + 'test_transfer_result_pred_0513.txt', language_model_scores_trans_pred)
    # transfer label
    compute_clinical_efficacy_scores(language_model_scores_trans_label, transfer_reports_label, ref_reports)
    compute_report_level_scores(language_model_scores_trans_label, transfer_reports_label, ref_reports)
    write_scores(path_to_write + 'test_transfer_result_label_0513.txt', language_model_scores_trans_label)


def calc_NLG_CE(path_to_read, path_to_write, gt_col, pred_col, is_clean=True, id_col =None, type = None):
    if path_to_read.endswith('.xlsx'):
        df = pd.read_excel(path_to_read, engine='openpyxl')
    elif path_to_read.endswith('.csv'):
        df = pd.read_csv(path_to_read)
    language_model_scores = create_language_model_scores_dict()

    if 'inputprompt' in pred_col:
        if pred_col in df.columns.values:
            ref_reports, pred_reports = df[gt_col].tolist(), df[pred_col].tolist()
        else:
            pred_col = pred_col[:-4] + '1500'
            ref_reports, pred_reports = df[gt_col].tolist(), df[pred_col].tolist()
    else:
        ref_reports, pred_reports = df[gt_col].tolist(), df[pred_col].tolist()
    # clean
    if is_clean:
        ref_reports_clean = [clean_report_mimic_cxr(x) for x in ref_reports]
        pred_reports_clean = [clean_report_mimic_cxr(x) for x in pred_reports]
    else:
        ref_reports_clean = ref_reports
        pred_reports_clean = pred_reports

    IDS = None
    if id_col is not None:
        IDS = df[id_col].tolist()
    compute_clinical_efficacy_scores(language_model_scores, pred_reports_clean, ref_reports_clean, IDS)
    compute_report_level_scores(language_model_scores, pred_reports_clean, ref_reports_clean)

    # os.makedirs(path_to_write, exist_ok = True)
    TYPE = pred_col if type is None else type
    root = path_to_write + f'{TYPE}_Metrics_NLG_CE.txt'
    write_scores(root, language_model_scores)

    return root



if __name__ == '__main__':
    # _ = calc_NLG_CE(os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))),
    #                 os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))),
    #                 'ground_truth',
    #                 'generated_text')


    _ = calc_NLG_CE(os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))),
                    os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))),
                    'reference_report_new',
                    'gpt_output_pred_clean',
                    # 'transfer_output_pred_clean',
                    type='lro')
    # _ = calc_NLG_CE(os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))),
    #                 os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))),
    #                 'reference_report',
    #                 'predict_report_R2Gen')
    # _ = calc_NLG_CE(os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))),
    #                 os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))),
    #                 'reference_report',
    #                 'predict_report')
    _ = calc_NLG_CE(os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))),
                    os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))),
                    'reference_report_new',
                    'predict_report_R2GenGPT_findings')
    # df_cmcrl = pd.read_excel(os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))),engine='openpyxl')
    # df_promptMRG = pd.read_excel(os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))), engine='openpyxl')
    # df = pd.merge(df_cmcrl, df_promptMRG, how='left', on='Path')
    # df.to_excel(os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))), engine='openpyxl')
    # df_r2g = pd.read_excel(os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))), engine='openpyxl')
    # base_score = calc_NLG_CE(os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))),
    #                     os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))),
    #                     'gt_report',
    #                     'r2gen_report')
    # pred10w_score = calc_NLG_CE(
    #     os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))),
    #     os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))),
    #     'gt_report',
    #     'gpt_output_pred_10w')
    # pred15w_score = calc_NLG_CE(
    #     os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))),
    #     os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))),
    #     'gt_report',
    #     'gpt_output_pred_15w')
    # label_score = calc_NLG_CE(
    #     os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))),
    #     os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))),
    #     'gt_report',
    #     'gpt_output_label')
    # tlabel_score = calc_NLG_CE(
    #     os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))),
    #     os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))),
    #     'gt_report',
    #     'transfer_output_label')
    # tpred15w_score = calc_NLG_CE(
    #     os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))),
    #     os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))),
    #     'gt_report',
    #     'transfer_output_pred_15w')
    # pred15w_score = calc_NLG_CE(
    #     os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))),
    #     os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))),
    #     'gt_report',
    #     'llm_infer_ABLATION_w_Diseaselabel_only')

    calc_NLG_CE(os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))),
                os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))),
                'reference_report',
                'predict_report')
    calc_NLG_CE(os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))),
                os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))),
                'reference_report_new',
                'predict_report')
    print()
    # translate_en_to_zh('The heart is moderately enlarged, consistent with cardiomegaly.')
     # df = pd.read_csv(os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))))
    # df = df[df['available'] == 'yes']
    # r2g_reports, ref_reports = df['r2gen_report'].tolist(), df['gt_report'].tolist()
    # compute_clinical_efficacy_scores(language_model_scores, r2g_reports, ref_reports)
    # compute_report_level_scores(language_model_scores, r2g_reports, ref_reports)
    # write_scores(os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))), language_model_scores)
    # df = df[df['gpt_output'] != 'blank']

    # analysis_report(os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))), os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))))

    # ROOT_PATH = os.environ.get('RESULT_XLSX', str(Path('out.xlsx')))
    # calc_NLG_CE_metrics(
    #     path_to_read=os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))),
    #     path_to_write=ROOT_PATH,
    #     use_clean=True, use_transfer=True)

    # calc_NLG_CE_metrics(
    #     path_to_read=os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))),
    #     path_to_write=ROOT_PATH + '3000_',
    #     use_clean=True, use_transfer=True)
    #
    # calc_NLG_CE_metrics(
    #     path_to_read=os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))),
    #     path_to_write=ROOT_PATH + '3500_',
    #     use_clean=True, use_transfer=True)

    # calc_NLG_CE_metrics(
    #     path_to_read=os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))),
    #     path_to_write=ROOT_PATH + 'final_',
    #     use_clean=True, use_transfer=True)

    ## clean
    # df = pd.read_csv(os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))))
    # # 对gt_report和gpt_output_label列应用clean_report_mimic_cxr函数
    # df['gpt_output_pred_clean'] = df['gpt_output_pred'].apply(clean_report_mimic_cxr)
    # df.to_csv(os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))), index=False)

    ### CLEAN with R2GEN
    # df_gpt_wlabel_clean = pd.read_excel(os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))), engine='openpyxl')
    # df_gpt_wpred = pd.read_csv(os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))))
    # df_gpt_wlabel_wpred = pd.merge(df_gpt_wlabel_clean, df_gpt_wpred[["id", "gpt_output"]], on="id", how="left")
    # df_gpt_wlabel_wpred['gpt_output_pred_clean'] = df_gpt_wlabel_wpred['gpt_output'].apply(clean_report_mimic_cxr)

    # df_gpt_wlabel_wpred = pd.read_excel(os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))), engine='openpyxl')
    # df_gpt_wlabel_wpred['r2gen_report_clean'] = df_gpt_wlabel_wpred['r2gen_report'].apply(clean_report_mimic_cxr)
    # df_gpt_wlabel_wpred.to_excel(os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))),engine='openpyxl')

    # df = pd.read_csv(os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))))
    # df['gpt_output_label_clean'] = df['gpt_output_label'].apply(clean_report_mimic_cxr)
    # df['gt_report_clean'] = df['all_reference_report'].apply(clean_report_mimic_cxr)
    # df.to_csv(os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))))
    # calc
    # df =  pd.read_excel(os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))), engine='openpyxl')
    # # df = pd.read_excel(os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))), engine='openpyxl')
    # language_model_scores_r2g, language_model_scores_gpt_label, language_model_scores_gpt_pred, language_model_scores_transfer  = \
    #     create_language_model_scores_dict(), create_language_model_scores_dict(), \
    #     create_language_model_scores_dict(), create_language_model_scores_dict()
    #
    #
    # language_model_scores_gpt_pred = create_language_model_scores_dict()
    # r2g_reports,  llm_reports_pred, ref_reports = df['r2gen_report'].tolist(), df['gpt_output_pred'].tolist(),  df['gt_report'].tolist()
    #
    # r2g_reports_clean, llm_reports_pred_clean, ref_reports_clean = df['r2gen_report_clean'].tolist(), df['gpt_output_pred_clean'].tolist(), df['gt_report_clean'].tolist()
    # r2g_reports_clean, llm_reports_label_clean, llm_reports_pred_clean, transfer_reports_clean, ref_reports_clean = df['r2gen_report_clean'].tolist(), df[
    #     'gpt_output_label_clean'].tolist(), df['gpt_output_pred_clean'].tolist(), df['transfer_output_clean'].tolist(), df['gt_report_clean'].tolist()

    # # r2g
    # compute_clinical_efficacy_scores(language_model_scores_r2g, r2g_reports_clean, ref_reports_clean)
    # compute_report_level_scores(language_model_scores_r2g, r2g_reports_clean, ref_reports_clean)
    # # llm_label
    # compute_clinical_efficacy_scores(language_model_scores_gpt_label, llm_reports_label_clean, ref_reports_clean)
    # compute_report_level_scores(language_model_scores_gpt_label, llm_reports_label_clean, ref_reports_clean)
    # # llm_pred
    # compute_clinical_efficacy_scores(language_model_scores_gpt_pred, llm_reports_pred_clean, ref_reports_clean)
    # compute_report_level_scores(language_model_scores_gpt_pred, llm_reports_pred_clean, ref_reports_clean)
    # transfer
    # compute_clinical_efficacy_scores(language_model_scores_transfer, transfer_reports_clean, ref_reports_clean)
    # compute_report_level_scores(language_model_scores_transfer, transfer_reports_clean, ref_reports_clean)


    # write_scores(os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))), language_model_scores_r2g)
    # write_scores(os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))), language_model_scores_gpt_label)
    # write_scores(os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))),language_model_scores_gpt_pred)
    # write_scores(os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))), language_model_scores_transfer)
    # # gpt NLG & CE results
    ## r2g
    # df_gpt = pd.read_csv(os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))))
    # language_model_scores_r2g_gpt, language_model_scores_gpt = create_language_model_scores_dict(), create_language_model_scores_dict()
    # r2g_reports_gpt, llm_reports_gpt, ref_reports_gpt = df_gpt['r2gen_report'].tolist(), df_gpt['gpt_output'].tolist(), df_gpt['gt_report'].tolist()
    # compute_clinical_efficacy_scores(language_model_scores_r2g_gpt, r2g_reports_gpt, ref_reports_gpt)
    # compute_report_level_scores(language_model_scores_r2g_gpt, r2g_reports_gpt, ref_reports_gpt)
    # write_scores(os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))), language_model_scores_r2g_gpt)
    # ## gpt-4-mini
    # compute_clinical_efficacy_scores(language_model_scores_gpt, llm_reports_gpt, ref_reports_gpt)
    # compute_report_level_scores(language_model_scores_gpt, llm_reports_gpt, ref_reports_gpt)
    # write_scores(os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))), language_model_scores_gpt)

    ## deepseek NLG & CE results
    # df_ds = pd.read_csv(os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))))
    # language_model_scores_ds = create_language_model_scores_dict()
    # llm_reports_ds_clean, ref_reports_clean = df_ds['deepseek_output_report_clean'].tolist(), df_ds['gt_report_clean'].tolist()
    # compute_report_level_scores(language_model_scores_ds, llm_reports_ds_clean, ref_reports_clean)
    # compute_clinical_efficacy_scores(language_model_scores_ds, llm_reports_ds_clean, ref_reports_clean)
    # write_scores(os.environ.get('RESULT_XLSX', str(Path('out.xlsx'))), language_model_scores_ds)



