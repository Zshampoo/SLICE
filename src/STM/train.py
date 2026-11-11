# train.py
import sys
import os
import random
import argparse
import numpy as np
import pandas as pd
import torch
import transformers
from tqdm import tqdm

from src.STM.eval_to_ts import label_input_report, rearrange_labels
from src.STM.src.config import Config
from src.STM.src.data_preprocessing import (
    load_and_split_dataset,
    preprocess_dataset_noprompt,
    preprocess_dataset_winputprompt,
)
from src.STM.src.model import load_model_and_tokenizer
from src.STM.src.trainer import build_trainer_weval
import openpyxl
import warnings
warnings.filterwarnings("ignore")

transformers.trainer_utils.tqdm = lambda *args, **kwargs: tqdm(*args, disable=True, **kwargs)


def set_seed(seed: int):
    """设置随机种子，确保训练过程的随机性固定"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用 GPU
    # 让 CUDA 的可重复性问题得到避免（提高训练的可重复性）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="少量样本试跑：验证数据流 + build_trainer_weval + NLG/F1/RadGraph 评估与检查点落盘")
    parser.add_argument("--smoke-train-n", type=int, default=64)
    parser.add_argument("--smoke-eval-n", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--eval-steps", type=int, default=None,
                        help="default from Config.eval_steps (1000)")
    parser.add_argument("--save-steps", type=int, default=None,
                        help="default from Config.save_steps (1000)")
    args, _ = parser.parse_known_args()

    # 设置固定的随机种子
    set_seed(42)

    output_dir = Config.output_dir
    if args.smoke:
        output_dir = os.path.join(output_dir, "smoke_test")

    os.makedirs(output_dir, exist_ok=True)

    print("Loading data...")
    train_ds, test_ds = load_and_split_dataset()

    if args.smoke:
        n_tr = min(args.smoke_train_n, len(train_ds))
        n_te = min(args.smoke_eval_n, len(test_ds))
        train_ds = train_ds.select(range(n_tr))
        test_ds = test_ds.select(range(n_te))
        print(f"[smoke] using train={n_tr}, eval={n_te}")

    print("Loading tokenizer and model...")
    model, tokenizer = load_model_and_tokenizer()

    use_prompt = getattr(Config, "use_prompt", True)
    if use_prompt:
        # [UNC] is required by wba prompts (uncertain); [POS]/[NEG]/[BLA] are the requested labels.
        tokenizer.add_tokens(['[POS]', '[NEG]', '[BLA]', '[UNC]'])
        model.resize_token_embeddings(len(tokenizer))
        print("Tokenizing data (with input prompt)...")
        test_ds = preprocess_dataset_winputprompt(test_ds, tokenizer, split="test")
        train_ds = preprocess_dataset_winputprompt(train_ds, tokenizer, split="train")
    else:
        print("Tokenizing data (NO prompt)...")
        test_ds = preprocess_dataset_noprompt(test_ds, tokenizer, split="test")
        train_ds = preprocess_dataset_noprompt(train_ds, tokenizer, split="train")

    print("Building trainer...")
    eval_steps_cfg = Config.eval_steps if args.eval_steps is None else args.eval_steps
    save_steps_cfg = Config.save_steps if args.save_steps is None else args.save_steps
    eval_steps = 2 if args.smoke else eval_steps_cfg
    save_steps = 2 if args.smoke else save_steps_cfg
    max_steps = 4 if args.smoke else args.max_steps
    num_epochs = 1 if args.smoke else None

    # Keep ALL checkpoint-{save_steps, 2*save_steps, ...} (no auto-delete).
    # Existing content under output_dir is not wiped by Trainer; only same-named
    # checkpoint-{step} folders would be overwritten when that step is reached again.
    print(f"output_dir={output_dir} (save_total_limit=None, keep every {save_steps}-step ckpt)")
    trainer = build_trainer_weval(
        model, tokenizer, train_ds, test_ds, output_dir,
        eval_steps=eval_steps,
        save_steps=save_steps,
        max_steps=max_steps,
        num_train_epochs=num_epochs,
        generation_max_length=128 if args.smoke else None,
        save_total_limit=None,
    )

    print("Training...")
    trainer.train()

    # Final weights ONLY under checkpoint-final/ (not dumped into output_dir root)
    final_dir = os.path.join(output_dir, "checkpoint-final")
    os.makedirs(final_dir, exist_ok=True)
    print(f"Saving final model → {final_dir}")
    trainer.save_model(final_dir)
    try:
        tokenizer.save_pretrained(final_dir)
    except Exception as e:
        print(f"[warn] tokenizer.save_pretrained failed: {e}")

    print("Evaluating on test set...")
    try:
        predictions = trainer.predict(test_ds).predictions
        decoded = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        print("\n--- Sample predictions ---")
        for i, text in enumerate(decoded[:5]):
            print(f"[{i}] {text}")
    except Exception as e:
        print(f"[warn] final predict/sample decode failed (training+evals already saved): {e}")

    if args.smoke:
        print(f"\n[smoke] done. Check metrics under: {output_dir}")


def extract_chexbert_labels():
    # df_train = pd.read_csv(Config.train_csv_file)
    # input_reports_clean  = df_train['gpt_output_label_clean'].tolist()
    # input_labels = label_input_report(input_reports_clean)
    # labels_new, prompt_all = rearrange_labels(input_labels)
    # df_train.insert(0, 'input_prompt', prompt_all)
    # df_train.insert(0, 'input_labels', labels_new)
    # df_train.to_csv(Config.train_csv_file, index=False)

    df_test = pd.read_excel(Config.test_csv_file)
    input_reports_clean = df_test['gpt_output_label_clean'].tolist()
    input_labels = label_input_report(input_reports_clean)
    labels_new, prompt_all = rearrange_labels(input_labels)
    df_test.insert(0, 'input_prompt', prompt_all)
    df_test.insert(0, 'input_labels', labels_new)
    df_test.to_excel(Config.test_csv_file, engine='openpyxl', index=False)


if __name__ == "__main__":
    # extract_chexbert_labels()
    main()
