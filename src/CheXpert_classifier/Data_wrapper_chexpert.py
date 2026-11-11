import collections
import json
import logging
import os
from pathlib import Path
import cv2
import numpy
import numpy as np
import pandas as pd
import torch
from PIL import Image
from albumentations.pytorch import ToTensorV2
from skimage import io
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, transforms
import albumentations as A
from torch.utils.data import DataLoader
# from model.lavis.data.ReportDataset import ExpandChannels

# from local_config import VIS_ROOT, PATH_TO_MIMIC_CXR

class Chexpert_Dataset(Dataset):
    def __init__(self, df, chexlabels, split = 'train', root_dir = None, shape = 224, label_type ='U-zero',loss_weighting="none", use_augs=False):

        super().__init__()

        self.df = df
        # load csv file
        self.label_cols = chexlabels
        self.label_type = label_type

        self.root_dir = root_dir if root_dir is not None else None
        self.split = split

        if shape == 224:
            self.vis_transforms = Compose( [Resize(224), CenterCrop(224), ToTensor(), ExpandChannels()])  # mimic batch*3*224*224 for swintransformer
        else:
            self.vis_transforms = Compose([Resize(512), CenterCrop(488), ToTensor(), ExpandChannels()])  # @Radialog:https://github.com/ChantalMP/RaDialog_v2/blob/master/findings_classifier/chexpert_dataset.py mimic batch*3*488*488 for resnet50

        # self.vis_transforms = transforms.Compose([
        #                                     transforms.Resize((224, 224)),
        #                                     ExpandChannels(),
        #                                     transforms.ToTensor(),
        #                                     transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))]) # iuxray

        # self.vis_transforms = Compose([Resize(512), CenterCrop(488), ToTensor()]) # batch*1*488*488
        # self.vis_transforms = A.Compose([A.Resize(512, 512, always_apply=True),
        #                                  A.CenterCrop(488, 488),
        #                                  ToTensorV2(),
        #                                  ExpandChannels()])
        if use_augs:
            aug_tfm = transforms.Compose([transforms.RandomAffine(degrees=30, shear=15),
                                          transforms.ColorJitter(brightness=0.2, contrast=0.2)]) # https://github.com/ChantalMP/RaDialog_v2/blob/master/findings_classifier/chexpert_dataset.py

            self.vis_transforms = transforms.Compose([self.vis_transforms, aug_tfm])
        self.loss_weighting = loss_weighting

    def get_class_weights(self):
        """Compute class weights based on the inverse of class frequencies.

        Returns:
            Dict[str, float]: Class weights.
        """
        if self.loss_weighting == "none":
            return torch.ones(len(self.label_cols), dtype=torch.float32)

        label_counts = torch.zeros(len(self.label_cols), dtype=torch.float32)
        # iterate over dataframe getting rows
        for _, ann in self.df.iterrows():
            chex_labels = self._extract_chex_labels_from_row(ann, self.label_type)
            label_counts += chex_labels

        # Compute class weights
        if self.loss_weighting == "lin":
            class_weights = len(self.df) / label_counts
        elif self.loss_weighting == "log":
            class_weights = torch.log(len(self.df) / label_counts)

        return class_weights

    def remap_to_uint8(self, array: np.ndarray, percentiles=None) -> np.ndarray:
        """Remap values in input so the output range is :math:`[0, 255]`.

        Percentiles can be used to specify the range of values to remap.
        This is useful to discard outliers in the input data.

        :param array: Input array.
        :param percentiles: Percentiles of the input values that will be mapped to ``0`` and ``255``.
            Passing ``None`` is equivalent to using percentiles ``(0, 100)`` (but faster).
        :returns: Array with ``0`` and ``255`` as minimum and maximum values.
        """
        array = array.astype(float)
        if percentiles is not None:
            len_percentiles = len(percentiles)
            if len_percentiles != 2:
                message = (
                    'The value for percentiles should be a sequence of length 2,'
                    f' but has length {len_percentiles}'
                )
                raise ValueError(message)
            a, b = percentiles
            if a >= b:
                raise ValueError(f'Percentiles must be in ascending order, but a sequence "{percentiles}" was passed')
            if a < 0 or b > 100:
                raise ValueError(f'Percentiles must be in the range [0, 100], but a sequence "{percentiles}" was passed')
            cutoff: np.ndarray = np.percentile(array, percentiles)
            array = np.clip(array, *cutoff)
        array -= array.min()
        array /= array.max()
        array *= 255
        return array.astype(np.uint8)

    def load_image(self, path) -> Image.Image:
        """Load an image from disk.

        The image values are remapped to :math:`[0, 255]` and cast to 8-bit unsigned integers.

        :param path: Path to image.
        :returns: Image as ``Pillow`` ``Image``.
        """
        # Although ITK supports JPEG and PNG, we use Pillow for consistency with older trained models
        if path.suffix in [".jpg", ".jpeg", ".png"]:
            image = io.imread(path)
        else:
            raise ValueError(f"Image type not supported, filename was: {path}")
        # if isinstance(image, np.ndarray):
        image = self.remap_to_uint8(image)
        return Image.fromarray(image).convert("L")

    def _extract_chex_labels_from_row(self, row: pd.Series, label_type='U-zero') -> torch.Tensor:
        if label_type == '4class':
            # CheXbert class ids. No Finding (name contains "No Finding") is binary:
            # blank=0, positive=1. The other 13 heads are
            # blank=0, positive=1, negative=2, uncertain=3.
            labels = torch.zeros(len(self.label_cols), dtype=torch.long)
            for i, col in enumerate(self.label_cols):
                val = row[col]
                is_nf = "No Finding" in str(col)
                if pd.isna(val):
                    labels[i] = 0
                    continue
                v = float(val)
                if is_nf:
                    labels[i] = 1 if v == 1.0 else 0
                elif v == 1.0:
                    labels[i] = 1
                elif v == 0.0:
                    labels[i] = 2
                elif v == -1.0:
                    labels[i] = 3
                else:
                    raise ValueError(f"unexpected label {col}={val}")
            return labels
        labels = torch.zeros(len(self.label_cols), dtype=torch.float32)
        for i, col in enumerate(self.label_cols):
            if label_type=='U-zero':
                if row[col] == 1:
                    labels[i] = 1
            elif label_type=='U-one':
                if row[col] == 1 or row[col] == -1:
                    labels[i] = 1
        return labels

    def __getitem__(self, index):
        ann = self.df.iloc[index]


        if self.root_dir is not None:
            path = ann['image_path'].replace('.jpg', '.png')
            pathstr = ''
            for str in path.split('/')[1:]:
                pathstr += str + '/'
            pathstr = self.split + '/' + pathstr[:-1]
            image_path = os.path.join(self.root_dir, pathstr)
        else:
            image_path = ann['mimic_image_file_path']  # MIMIC path
        # image = self.load_image(Path(image_path)) # 报错

        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        # img_pil = Image.fromarray(image)
        img_pil_uint8 = self.remap_to_uint8(image)
        img_pil = Image.fromarray(img_pil_uint8).convert("L")

        image = self.vis_transforms(img_pil) # (3, 488, 488)

        # MIMIC uses all_reference_report; CheXpert+ annotation uses report
        if "report" in ann.index and pd.notna(ann.get("report", None)):
            report = ann["report"]
        else:
            report = ann.get("all_reference_report", "")

        label = self._extract_chex_labels_from_row(ann, self.label_type)

        sample = { "image": image,
                   "label": label,
                   "report": report}
        return sample

    def __len__(self):
        return len(self.df)


class Chexpert_Dataset_test(Dataset):
    def __init__(self, df, chexlabels, split = 'test', root_dir = None, label_type = 'U-zero', loss_weighting="none", use_augs=False):

        super().__init__()

        self.df = df
        # load csv file
        self.label_cols = chexlabels
        self.label_type = label_type
        self.root_dir = root_dir
        self.split = split

        self.vis_transforms = Compose([Resize(512), CenterCrop(488), ToTensor(), ExpandChannels()]) # batch*3*488*488

        # self.vis_transforms = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])  # iuxray

        # self.vis_transforms = Compose([Resize(512), CenterCrop(488), ToTensor()]) # batch*1*488*488
        # self.vis_transforms = A.Compose([A.Resize(512, 512, always_apply=True),
        #                                  A.CenterCrop(488, 488),
        #                                  ToTensorV2(),
        #                                  ExpandChannels()])
        if use_augs:
            aug_tfm = transforms.Compose([transforms.RandomAffine(degrees=30, shear=15),
                                          transforms.ColorJitter(brightness=0.2, contrast=0.2)])

            self.vis_transforms = transforms.Compose([self.vis_transforms, aug_tfm])
        self.loss_weighting = loss_weighting

    def get_class_weights(self):
        """Compute class weights based on the inverse of class frequencies.

        Returns:
            Dict[str, float]: Class weights.
        """
        if self.loss_weighting == "none":
            return torch.ones(len(self.label_cols), dtype=torch.float32)

        label_counts = torch.zeros(len(self.label_cols), dtype=torch.float32)
        # iterate over dataframe getting rows
        for _, ann in self.df.iterrows():
            chex_labels = self._extract_chex_labels_from_row(ann,  self.label_type)
            label_counts += chex_labels

        # Compute class weights
        if self.loss_weighting == "lin":
            class_weights = len(self.df) / label_counts
        elif self.loss_weighting == "log":
            class_weights = torch.log(len(self.df) / label_counts)

        return class_weights

    def remap_to_uint8(self, array: np.ndarray, percentiles=None) -> np.ndarray:
        """Remap values in input so the output range is :math:`[0, 255]`.

        Percentiles can be used to specify the range of values to remap.
        This is useful to discard outliers in the input data.

        :param array: Input array.
        :param percentiles: Percentiles of the input values that will be mapped to ``0`` and ``255``.
            Passing ``None`` is equivalent to using percentiles ``(0, 100)`` (but faster).
        :returns: Array with ``0`` and ``255`` as minimum and maximum values.
        """
        array = array.astype(float)
        if percentiles is not None:
            len_percentiles = len(percentiles)
            if len_percentiles != 2:
                message = (
                    'The value for percentiles should be a sequence of length 2,'
                    f' but has length {len_percentiles}'
                )
                raise ValueError(message)
            a, b = percentiles
            if a >= b:
                raise ValueError(f'Percentiles must be in ascending order, but a sequence "{percentiles}" was passed')
            if a < 0 or b > 100:
                raise ValueError(f'Percentiles must be in the range [0, 100], but a sequence "{percentiles}" was passed')
            cutoff: np.ndarray = np.percentile(array, percentiles)
            array = np.clip(array, *cutoff)
        array -= array.min()
        array /= array.max()
        array *= 255
        return array.astype(np.uint8)

    def load_image(self, path) -> Image.Image:
        """Load an image from disk.

        The image values are remapped to :math:`[0, 255]` and cast to 8-bit unsigned integers.

        :param path: Path to image.
        :returns: Image as ``Pillow`` ``Image``.
        """
        # Although ITK supports JPEG and PNG, we use Pillow for consistency with older trained models
        if path.suffix in [".jpg", ".jpeg", ".png"]:
            image = io.imread(path)
        else:
            raise ValueError(f"Image type not supported, filename was: {path}")
        # if isinstance(image, np.ndarray):
        image = self.remap_to_uint8(image)
        return Image.fromarray(image).convert("L")

    def _extract_chex_labels_from_row(self, row: pd.Series, label_type = 'U-zero') -> torch.Tensor:
        labels = torch.zeros(len(self.label_cols), dtype=torch.float32)
        for i, col in enumerate(self.label_cols):
            if label_type == 'U-zero':
                if row[col] == 1: # U-zeros
                    labels[i] = 1
            elif label_type == 'U-one':
                if row[col] == 1 or row[col] == -1: # U-ones
                    labels[i] = 1
        return labels

    def get_ap_imagepath_from_iuxray(self, id):
        with open(str(Path(__import__('os').environ.get('DATA_ROOT', './data'))), 'r') as f:
            json_iux = json.load(f)
        for item in json_iux:
            if item['id'] == id:
                return item['image_path'][0]

    def get_imagepath_from_chexpertplus(self, ann):
        if self.root_dir is not None:
            path = ann['image_path'].replace('.jpg', '.png')
            pathstr = ''
            for str in path.split('/')[1:]:
                pathstr += str + '/'
            pathstr = self.split + '/' + pathstr[:-1]
            image_path = os.path.join(self.root_dir, pathstr)
            return image_path

    def __getitem__(self, index):
        ann = self.df.iloc[index]

        # image_path = ann['mimic_image_file_path'] # mimic
        # image_dir = './data/<path>' + ann['Path'] # iuxray
        # image_dir = './data/<path>' + ann['Path']  # iuxray

        image_path = self.get_imagepath_from_chexpertplus(ann) # chexpertplus

        # image = self.load_image(Path(image_path)) # 报错

        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) # mimic

        # iuxray
        # pa_img_path = self.get_ap_imagepath_from_iuxray(ann['Path'])
        # pa_img_path = os.path.join(image_dir, os.listdir(image_dir)[0])
        # image_pa = Image.open(os.path.join(image_dir, pa_img_path)).convert('RGB')
        # if self.vis_transforms is not None:
        #     image = self.vis_transforms(image)

        # img_pil = Image.fromarray(image)
        img_pil_uint8 = self.remap_to_uint8(image)
        img_pil = Image.fromarray(img_pil_uint8).convert("L")

        image = self.vis_transforms(img_pil) # (3, 488, 488)

        # report = ann['all_reference_report'] # mimic
        report = ann['report'] # chexpertplus

        # report = ann['reference_report']

        label = self._extract_chex_labels_from_row(ann, self.label_type)

        sample = { "image": image,
                   "label": label,
                   "report": report,
                   "file_id":ann['image_path']}
        return sample

    def __len__(self):
        return len(self.df)

class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())

        self.examples = self.ann[self.split]
        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

    def __len__(self):
        return len(self.examples)

class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        report= example['report']
        label = None
        # report_ids = example['ids']
        # report_masks = example['mask']
        # seq_length = len(report_ids)
        sample = {"image": image,
                  "label": label,
                  "report": report,
                  "file_id": image_id}
        return sample

class ExpandChannels:
    """
    Transforms an image with one channel to an image with three channels by copying
    pixel intensities of the image along the 1st dimension.
    """

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """
        :param data: Tensor of shape [1, H, W].
        :return: Tensor with channel copied three times, shape [3, H, W].
        """
        if data.shape[0] != 1:
            raise ValueError(f"Expected input of shape [1, H, W], found {data.shape}")
        return torch.repeat_interleave(data, 3, dim=0)

def get_dataframes_train(path, select = 'chexBert'):
    usecols_chexPert = ["mimic_image_file_path",
                        "all_reference_report",
                        "No Finding_x",
                        "Enlarged Cardiomediastinum_x",
                        "Cardiomegaly_x",
                        "Lung Opacity_x",
                        "Lung Lesion_x",
                        "Edema_x",
                        "Consolidation_x",
                        "Pneumonia_x",
                        "Atelectasis_x",
                        "Pneumothorax_x",
                        "Pleural Effusion_x",
                        "Pleural Other_x",
                        "Fracture_x",
                        "Support Devices_x"]

    usecols_chexBert = ["mimic_image_file_path",
                        "all_reference_report",
                        "No Finding_y",
                        "Enlarged Cardiomediastinum_y",
                        "Cardiomegaly_y",
                        "Lung Opacity_y",
                        "Lung Lesion_y",
                        "Edema_y",
                        "Consolidation_y",
                        "Pneumonia_y",
                        "Atelectasis_y",
                        "Pneumothorax_y",
                        "Pleural Effusion_y",
                        "Pleural Other_y",
                        "Fracture_y",
                        "Support Devices_y"]

    usecols_chexpert_plus = ["image_path",
                        "report",
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

    chexlabels = None
    if select == 'chexBert':
        usecols = usecols_chexBert
        logging.info(f"Using chexBert labels")
        chexlabels = usecols_chexBert[2:]
    elif select == 'chexPert':
        usecols = usecols_chexPert
        logging.info(f"Using chexPert labels")
        chexlabels = usecols_chexPert[2:]
    elif select == 'chexpert_plus':
        usecols = usecols_chexpert_plus
        logging.info(f"Using chexpert_plus labels")
        chexlabels = usecols_chexpert_plus[2:]
    else:
        logging.info(f"Select {select} error!")

    datasets_as_dfs = pd.read_excel(path, usecols=usecols, engine='openpyxl')
    return datasets_as_dfs, chexlabels

def get_dataframes_test(path, select = 'chexBert'):
    usecols_chexPert = ["mimic_image_file_path",
                        "all_reference_report",
                        "chexpert_full_zero_or_blank_or_uncertain",
                        "No Finding_x",
                        "Enlarged Cardiomediastinum_x",
                        "Cardiomegaly_x",
                        "Lung Opacity_x",
                        "Lung Lesion_x",
                        "Edema_x",
                        "Consolidation_x",
                        "Pneumonia_x",
                        "Atelectasis_x",
                        "Pneumothorax_x",
                        "Pleural Effusion_x",
                        "Pleural Other_x",
                        "Fracture_x",
                        "Support Devices_x"]

    usecols_chexBert = ["mimic_image_file_path",
                        "all_reference_report",
                        # "chexbert_full_zero_or_blank_or_uncertain",
                        "available",
                        "No Finding_y",
                        "Enlarged Cardiomediastinum_y",
                        "Cardiomegaly_y",
                        "Lung Opacity_y",
                        "Lung Lesion_y",
                        "Edema_y",
                        "Consolidation_y",
                        "Pneumonia_y",
                        "Atelectasis_y",
                        "Pneumothorax_y",
                        "Pleural Effusion_y",
                        "Pleural Other_y",
                        "Fracture_y",
                        "Support Devices_y"]

    usecols_iuxray = [  "Path",
                        "reference_report",
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
                        "Support Devices",
                        "No Finding"]
    usecols_chexpert_plus = ["image_path",
                             "report",
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

    chexlabels = None
    if select == 'chexBert':
        usecols = usecols_chexBert
        logging.info(f"Using chexBert labels")
        chexlabels = usecols_chexBert[3:]
    elif select == 'chexPert':
        usecols = usecols_chexPert
        logging.info(f"Using chexPert labels")
        chexlabels = usecols_chexPert[3:]
    elif select == 'iuxray':
        usecols = usecols_iuxray
        logging.info(f"Using iuxray labels (uone)")
        chexlabels = usecols_iuxray[2:]
    elif select == 'chxepertplus':
        usecols = usecols_chexpert_plus
        logging.info(f"Using chxepertplus labels (uone)")
        chexlabels = usecols_chexpert_plus[2:]
    else:
        logging.info(f"Select {select} error!")

    datasets_as_dfs = pd.read_excel(path, usecols=usecols, engine='openpyxl')
    return datasets_as_dfs, chexlabels

def label_chexbert():
    df_chexbert = pd.read_excel(str(Path(__import__('os').environ.get('DATA_ROOT', './data'))), engine='openpyxl')
    # df_train_phase1 = pd.read_excel('./data/<path>', engine='openpyxl')
    # df_valid = pd.read_excel('./data/<path>', engine='openpyxl')
    df_test = pd.read_excel(str(Path(__import__('os').environ.get('DATA_ROOT', './data'))), engine='openpyxl')
    df_train_phase2 = pd.read_excel(str(Path(__import__('os').environ.get('DATA_ROOT', './data'))), engine='openpyxl')
    # df_train_phase1 = pd.merge(df_train_phase1, df_chexbert, on =['dicom_id'], how='left')
    # df_valid = pd.merge(df_valid, df_chexbert, on=['dicom_id'], how='left')
    df_test = pd.merge(df_test, df_chexbert, on=['dicom_id'], how='left')
    df_train_phase2 = pd.merge(df_train_phase2, df_chexbert, on=['dicom_id'], how='left')
    # df_train_phase1.to_excel('./data/<path>')
    # df_valid.to_excel('./data/<path>')
    df_test.to_excel(str(Path(__import__('os').environ.get('DATA_ROOT', './data'))))
    df_train_phase2.to_excel(str(Path(__import__('os').environ.get('DATA_ROOT', './data'))))


class Chexpert_5_200_testset(Dataset):
    def __init__(self, df, chexlabels, label_type = 'U-zero'):

        super().__init__()

        self.df = df
        # load csv file
        self.label_cols = chexlabels
        self.label_type = label_type

        self.vis_transforms = Compose([Resize(512), CenterCrop(488), ToTensor(), ExpandChannels()]) # batch*3*488*488
        # self.vis_transforms = Compose([Resize(512), CenterCrop(488), ToTensor()]) # batch*1*488*488
        # self.vis_transforms = A.Compose([A.Resize(512, 512, always_apply=True),
        #                                  A.CenterCrop(488, 488),
        #                                  ToTensorV2(),
        #                                  ExpandChannels()])

    def remap_to_uint8(self, array: np.ndarray, percentiles=None) -> np.ndarray:
        """Remap values in input so the output range is :math:`[0, 255]`.

        Percentiles can be used to specify the range of values to remap.
        This is useful to discard outliers in the input data.

        :param array: Input array.
        :param percentiles: Percentiles of the input values that will be mapped to ``0`` and ``255``.
            Passing ``None`` is equivalent to using percentiles ``(0, 100)`` (but faster).
        :returns: Array with ``0`` and ``255`` as minimum and maximum values.
        """
        array = array.astype(float)
        if percentiles is not None:
            len_percentiles = len(percentiles)
            if len_percentiles != 2:
                message = (
                    'The value for percentiles should be a sequence of length 2,'
                    f' but has length {len_percentiles}'
                )
                raise ValueError(message)
            a, b = percentiles
            if a >= b:
                raise ValueError(f'Percentiles must be in ascending order, but a sequence "{percentiles}" was passed')
            if a < 0 or b > 100:
                raise ValueError(f'Percentiles must be in the range [0, 100], but a sequence "{percentiles}" was passed')
            cutoff: np.ndarray = np.percentile(array, percentiles)
            array = np.clip(array, *cutoff)
        array -= array.min()
        array /= array.max()
        array *= 255
        return array.astype(np.uint8)

    def load_image(self, path) -> Image.Image:
        """Load an image from disk.

        The image values are remapped to :math:`[0, 255]` and cast to 8-bit unsigned integers.

        :param path: Path to image.
        :returns: Image as ``Pillow`` ``Image``.
        """
        # Although ITK supports JPEG and PNG, we use Pillow for consistency with older trained models
        if path.suffix in [".jpg", ".jpeg", ".png"]:
            image = io.imread(path)
        else:
            raise ValueError(f"Image type not supported, filename was: {path}")
        # if isinstance(image, np.ndarray):
        image = self.remap_to_uint8(image)
        return Image.fromarray(image).convert("L")

    def _extract_chex_labels_from_row(self, row: pd.Series, label_type = 'U-zero') -> torch.Tensor:
        labels = torch.zeros(len(self.label_cols), dtype=torch.float32)
        for i, col in enumerate(self.label_cols):
            if label_type == 'U-zero':
                if row[col] == 1: # U-zeros
                    labels[i] = 1
            elif label_type == 'U-one':
                if row[col] == 1 or row[col] == -1: # U-ones
                    labels[i] = 1
        return labels

    def __getitem__(self, index):
        ann = self.df.iloc[index]

        image_path = ann['Path']

        # image = self.load_image(Path(image_path)) # 报错

        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        # img_pil = Image.fromarray(image)
        img_pil_uint8 = self.remap_to_uint8(image)
        img_pil = Image.fromarray(img_pil_uint8).convert("L")

        image = self.vis_transforms(img_pil) # (3, 488, 488)

        # report = ann['all_reference_report']

        label = self._extract_chex_labels_from_row(ann, self.label_type)

        sample = { "image": image,
                   "label": label,
                   "file_id":image_path}
        return sample

    def __len__(self):
        return len(self.df)

if __name__ == '__main__':

    label_chexbert()
