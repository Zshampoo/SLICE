# src/data_preprocessing.py
import ast

import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import BartTokenizer
from src.ECG.src.config import Config

label_cols = ['No Finding_y',
              'Enlarged Cardiomediastinum_y',
              'Cardiomegaly_y',
              'Lung Opacity_y',
              'Lung Lesion_y',
              'Edema_y',
              'Consolidation_y',
              'Pneumonia_y',
              'Atelectasis_y',
              'Pneumothorax_y',
              'Pleural Effusion_y',
              'Pleural Other_y',
              'Fracture_y',
              'Support Devices_y']

SCORES_NP = {0.0: '[NEG]', 1.0: '[POS]'}
POLARITY_TO_TOKEN = {
    'Positive': '[POS]',
    'positive': '[POS]',
    'Negative': '[NEG]',
    'negative': '[NEG]',
}


def _read_table_flexible(path: str, sheet=None) -> pd.DataFrame:
    """Read CSV or Excel with encoding fallbacks for CSV."""
    path_l = str(path).lower()
    if path_l.endswith((".xlsx", ".xls")):
        kwargs = {"engine": "openpyxl"}
        if sheet is not None:
            kwargs["sheet_name"] = sheet
        return pd.read_excel(path, **kwargs)
    last_err = None
    for enc in ("utf-8", "utf-8-sig", "latin1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception as e:
            last_err = e
    raise last_err


def _read_csv_flexible(path: str) -> pd.DataFrame:
    """Backward-compatible alias."""
    return _read_table_flexible(path)


def disease_list_to_prompt(disease_info) -> str:
    """Prompt string ending with a trailing space.

    Accepts either:
      - already-tokenized lists / strings: ['[BLA]', '[POS]', '[NEG]', '[UNC]', ...]
      - polarity lists: [['No Finding','Negative'], ...] (optional score)
    """
    if disease_info is None or (isinstance(disease_info, float) and pd.isna(disease_info)):
        return ""
    if isinstance(disease_info, str):
        s = disease_info.strip()
        if not s:
            return ""
        # Already "[BLA] [POS] ..." (not a Python list literal).
        if s.startswith("[") and not (s.startswith("['") or s.startswith('["')):
            return s if s.endswith(" ") else s + " "
        disease_info = ast.literal_eval(s)
    if isinstance(disease_info, (list, tuple)) and disease_info:
        first = disease_info[0]
        if isinstance(first, str) and first.startswith("["):
            return " ".join(str(t) for t in disease_info) + " "
    tokens = []
    for item in disease_info:
        polarity = item[1]
        token = POLARITY_TO_TOKEN.get(polarity)
        if token is None:
            # fallback: treat unknown / uncertain as negative
            token = '[NEG]'
        tokens.append(token)
    return ' '.join(tokens) + ' '


def load_and_split_dataset():
    """Load train/test tables with Config column mapping.

    If Config.use_prompt is False, only input/output columns are required
    (no disease prompt). Drops rows with algo3_n_paired_components == 0 when present.
    """
    train_sheet = getattr(Config, "train_sheet", None)
    test_sheet = getattr(Config, "test_sheet", None)
    train_df = _read_table_flexible(Config.train_csv_file, sheet=train_sheet)
    test_df = _read_table_flexible(Config.test_csv_file, sheet=test_sheet)

    use_prompt = getattr(Config, "use_prompt", True)
    train_prompt = Config.train_prompt_col
    train_input = Config.train_input_col
    train_output = Config.train_output_col
    train_n_col = Config.train_n_comp_col
    test_prompt = Config.test_prompt_col
    test_input = Config.test_input_col
    test_output = Config.test_output_col

    train_need = [train_input, train_output] + ([train_prompt] if use_prompt and train_prompt else [])
    test_need = [test_input, test_output] + ([test_prompt] if use_prompt and test_prompt else [])
    train_cols = list(train_need)
    test_cols = list(test_need)
    if Config.train_id_col in train_df.columns:
        train_cols = [Config.train_id_col] + train_cols
    if train_n_col and train_n_col in train_df.columns:
        train_cols = train_cols + [train_n_col]
    if Config.test_id_col in test_df.columns:
        test_cols = [Config.test_id_col] + test_cols

    missing_train = [c for c in train_need if c not in train_df.columns]
    missing_test = [c for c in test_need if c not in test_df.columns]
    if missing_train:
        raise KeyError(f"train missing columns: {missing_train}")
    if missing_test:
        raise KeyError(f"test missing columns: {missing_test}")

    n_raw = len(train_df)
    n_zero = 0
    if train_n_col and train_n_col in train_df.columns:
        n_zero = int((train_df[train_n_col] == 0).sum())
        train_df = train_df[train_df[train_n_col] != 0].copy()
    train_df = train_df.dropna(subset=train_need).copy()
    test_df = test_df.dropna(subset=test_need).copy()

    train_df[train_input] = train_df[train_input].astype(str)
    train_df[train_output] = train_df[train_output].astype(str)
    test_df[test_input] = test_df[test_input].astype(str)
    test_df[test_output] = test_df[test_output].astype(str)

    print("=" * 72)
    print("Dataset / hyper-parameters")
    print("=" * 72)
    print(f"train_file: {Config.train_csv_file} sheet={train_sheet}")
    print(f"test_file:  {Config.test_csv_file} sheet={test_sheet}")
    print(f"output_dir: {Config.output_dir}")
    print(f"use_prompt: {use_prompt}")
    print(f"train columns: prompt={train_prompt} | input={train_input} | output={train_output}")
    print(f"test  columns: prompt={test_prompt} | input={test_input} | output={test_output}")
    print(
        f"train samples: {len(train_df)}  (raw={n_raw}, dropped n_comp==0: {n_zero})"
    )
    print(f"test  samples: {len(test_df)}")
    print(
        f"batch_size train/eval: {Config.train_batch_size}/{Config.eval_batch_size} | "
        f"epochs={Config.num_train_epochs} | lr={Config.learning_rate} | "
        f"max_in/out={Config.max_input_length}/{Config.max_target_length} | "
        f"eval/save_steps={Config.eval_steps}/{Config.save_steps}"
    )
    print(f"model: {Config.model_name}")
    if train_n_col and train_n_col in train_df.columns:
        vc = train_df[train_n_col].value_counts().sort_index()
        print("Train algo3_n_paired_components distribution:")
        for k, v in vc.items():
            print(f"  n_comp={k}: {int(v)} ({100.0 * v / len(train_df):.2f}%)")
    steps_per_epoch = (len(train_df) + Config.train_batch_size - 1) // Config.train_batch_size
    total_steps = steps_per_epoch * Config.num_train_epochs
    print(f"approx total steps: {steps_per_epoch} × {Config.num_train_epochs} = {total_steps}")
    print("=" * 72)

    train_ds = Dataset.from_pandas(train_df[train_cols], preserve_index=False)
    test_ds = Dataset.from_pandas(test_df[test_cols], preserve_index=False)
    return train_ds, test_ds


def load_and_split_dataset_input_prompt():
    train_df = pd.read_csv(Config.train_csv_file)
    # train_df = pd.read_excel(Config.test_csv_file) # debug
    test_df = pd.read_excel(Config.test_csv_file)
    # test_df = test_df[:80]

    print(f'Loaded train: {len(train_df)} samples and test: {len(test_df)} samples')
    train_df['input_prompt'] = train_df['input_prompt'].apply(ast.literal_eval)
    test_df['input_prompt'] = test_df['input_prompt'].apply(ast.literal_eval)
    train_ds = Dataset.from_pandas(train_df[["gpt_output_label_clean", "gt_report_clean",
                                             'input_prompt']])
    test_ds = Dataset.from_pandas(test_df[["gpt_output_label_clean", "gt_report_clean",
                                           'input_prompt']])

    return train_ds, test_ds

def preprocess_dataset_woutputprompt(dataset, tokenizer: BartTokenizer, is_mask=True):
    def tokenize_fn(example):

        model_inputs = tokenizer(
            example["gpt_output_label_clean"],
            max_length=Config.max_input_length,
            truncation=True,
        )
        # build prompt
        labels = [example[col] for col in label_cols]
        prompt = ''
        prompt = [SCORES_NP[l] for l in labels]
        prompt = ' '.join(prompt) + ' '
        prompt_len = len(tokenizer(prompt).input_ids)-1

        # target = prompt + report
        target_text = prompt + example["gt_report_clean"]
        target_encoding = tokenizer(
            target_text,
            max_length=Config.max_input_length,
            truncation=True,
        )

        target_encoding["input_ids"][0] = tokenizer.bos_token_id
        labels = target_encoding["input_ids"]

        if is_mask:  # mask prompt tokens with -100 (ignored in loss)
            for i in range(prompt_len):
                if i < len(labels):
                    labels[i] = -100

        model_inputs["labels"] = labels
        return model_inputs

    return dataset.map(tokenize_fn, remove_columns=["gpt_output_label_clean", "gt_report_clean"])

def preprocess_dataset_winputprompt(dataset, tokenizer: BartTokenizer, split: str = "auto"):
    """
    Encoder input = 14*[POS/NEG] prompt + source report; labels = target report.

    Train:
      prompt  = label_disease_paired_components_full
      source  = llm_report_label_paired_components_full   (was gpt_output_label_clean)
      target  = reference_report_paired_components_full   (was gt_report_clean)
    Test:
      prompt  = pred_disease_info_chexber_uone_15w_knn_new
      source  = llm_report_changeprompt_disease_only_all  (was gpt_output_label_clean)
      target  = reference_report_new                      (was gt_report_clean)
    """
    cols = set(dataset.column_names)
    if split == "auto":
        if Config.train_prompt_col in cols and Config.train_input_col in cols:
            split = "train"
        elif Config.test_prompt_col in cols:
            split = "test"
        else:
            raise ValueError(f"Cannot infer split from columns: {dataset.column_names}")

    if split == "train":
        prompt_col = Config.train_prompt_col
        source_col = Config.train_input_col   # replaces gpt_output_label_clean
        target_col = Config.train_output_col  # replaces gt_report_clean
        remove_cols = [
            source_col,
            target_col,
            prompt_col,
            Config.train_id_col,
            Config.train_n_comp_col,
        ]
    elif split == "test":
        prompt_col = Config.test_prompt_col
        source_col = Config.test_input_col    # replaces gpt_output_label_clean
        target_col = Config.test_output_col   # replaces gt_report_clean
        remove_cols = [
            source_col,
            target_col,
            prompt_col,
            Config.test_id_col,
        ]
    else:
        raise ValueError(f"Unknown split={split}")

    def tokenize_fn(example):
        # 14 [POS]/[NEG] tokens from disease list, then source report
        prompt = disease_list_to_prompt(example[prompt_col])
        input_text = prompt + str(example[source_col])

        model_inputs = tokenizer(
            input_text,
            max_length=Config.max_input_length,
            truncation=True,
        )

        labels = tokenizer(
            str(example[target_col]),
            max_length=Config.max_target_length,
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    remove_cols = [c for c in remove_cols if c in dataset.column_names]
    return dataset.map(tokenize_fn, remove_columns=remove_cols)

def preprocess_dataset_w_bartdecoderinput(dataset, tokenizer: BartTokenizer):
    def tokenize_fn(example):
        # 1) encode source text (BART encoder input)
        source_encoding = tokenizer(
            example["gpt_output_label_clean"],
            max_length=Config.max_input_length,
            truncation=True,
        )

        # 2) build decoder input text (prompt + report)
        labels_raw = [example[col] for col in label_cols]  # e.g. ["cardiomegaly", "effusion"]
        prompt_text = ' '.join([SCORES_NP[lbl] for lbl in labels_raw]) + ' '  # e.g. "[POS] [NEG] [NEG] ..."
        report_text = example["gt_report_clean"]
        decoder_text = prompt_text + report_text

        ## attempt 1: insert bos between prompt and report
        decoder_encoding = tokenizer(
            decoder_text,
            max_length=Config.max_target_length,
            truncation=True,
            add_special_tokens=False  # do not prepend bos; place it between prompt and report
        )

        decoder_input_ids = decoder_encoding["input_ids"]

        prompt_len = len(tokenizer(prompt_text, add_special_tokens=False).input_ids)  # 14

        # 4) build labels (first prompt_len positions and padding -> -100)
        labels = []
        for i, token_id in enumerate(decoder_input_ids):
            if i < prompt_len:
                labels.append(-100)
            else:
                labels.append(token_id)

        decoder_input_ids_shifted = []
        for i in range(len(labels)):
            if i < prompt_len:  # keep prompt positions
                decoder_input_ids_shifted.append(decoder_input_ids[i])
            elif i == prompt_len:  # insert bos here
                decoder_input_ids_shifted.append(tokenizer.bos_token_id)
            else:  # shift report tokens right
                decoder_input_ids_shifted.append(decoder_input_ids[i-1])

        return {
            "input_ids": source_encoding["input_ids"],
            "labels": labels,
            "decoder_input_ids": decoder_input_ids_shifted,
        }

    return dataset.map(tokenize_fn, remove_columns=["gpt_output_label_clean", "gt_report_clean"])

def preprocess_dataset(dataset, tokenizer: BartTokenizer):
    def tokenize_fn(example):
        model_inputs = tokenizer(
            example["gpt_output_label_clean"],
            max_length=Config.max_input_length,
            truncation=True,
        )

        labels = tokenizer(
            example["gt_report_clean"],
            max_length=Config.max_target_length,
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(tokenize_fn, remove_columns=["gpt_output_label_clean", "gt_report_clean"])


def preprocess_dataset_noprompt(dataset, tokenizer: BartTokenizer, split: str = "auto"):
    """Encoder input = source report only (no [POS]/[NEG] disease prompt)."""
    cols = set(dataset.column_names)
    if split == "auto":
        if Config.train_input_col in cols:
            split = "train"
        elif Config.test_input_col in cols:
            split = "test"
        else:
            raise ValueError(f"Cannot infer split from columns: {dataset.column_names}")

    if split == "train":
        source_col = Config.train_input_col
        target_col = Config.train_output_col
        remove_cols = [source_col, target_col, Config.train_id_col, Config.train_n_comp_col]
    elif split == "test":
        source_col = Config.test_input_col
        target_col = Config.test_output_col
        remove_cols = [source_col, target_col, Config.test_id_col]
    else:
        raise ValueError(f"Unknown split={split}")

    def tokenize_fn(example):
        model_inputs = tokenizer(
            str(example[source_col]),
            max_length=Config.max_input_length,
            truncation=True,
        )
        labels = tokenizer(
            str(example[target_col]),
            max_length=Config.max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    remove_cols = [c for c in remove_cols if c in dataset.column_names]
    return dataset.map(tokenize_fn, remove_columns=remove_cols)
