#!/usr/bin/env python3
"""Train sentence→29-region {blank,normal,abnormal} classifier (CheXbert-style heads).

Uses train_phase1_wodup for train; stratified 2700 from train_phase2_wodup for val.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer

# allow `python train_sentence_region_bert.py` from this dir / rgrg root
RGRG_ROOT = Path(__file__).resolve().parents[3]
if str(RGRG_ROOT) not in sys.path:
    sys.path.insert(0, str(RGRG_ROOT))

from src.CheXbert.src.models.bert_labeler import (  # noqa: E402
    REGION_LABEL2ID,
    REGION_NAMES_29,
    bert_labeler,
)

XLSX_DEFAULT = str(Path(__import__('os').environ.get('DATA_ROOT', './data')))
BLUE_BERT = str(Path(__import__('os').environ.get('DATA_ROOT', './data')))
OUT_DEFAULT = str(Path(__import__('os').environ.get('DATA_ROOT', './data')))
VAL_COL = "is_val_BLUE_BERT_CLS"
TEXT_COL = "sentence_clean_FINDINGS_IMPRESSION"
N_VAL = 2700
PAD_IDX = 0


def normalize_label(v) -> int:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return REGION_LABEL2ID["blank"]
    s = str(v).strip().lower()
    if s not in REGION_LABEL2ID:
        raise ValueError(f"Unknown label: {v!r}")
    return REGION_LABEL2ID[s]


def sentence_bucket(lab_row: np.ndarray) -> str:
    """Stratification key to mirror train class mix: all_blank / abn_any / nor_only."""
    has_abn = (lab_row == 2).any()
    has_nor = (lab_row == 1).any()
    if has_abn:
        return "abn_any"
    if has_nor:
        return "nor_only"
    return "all_blank"


def load_sheet_labels(xlsx: str, sheet: str):
    print(f"Reading sheet {sheet} ...", flush=True)
    df = pd.read_excel(xlsx, sheet_name=sheet, engine="openpyxl")
    missing = [c for c in REGION_NAMES_29 if c not in df.columns]
    if missing:
        raise KeyError(f"Missing region cols on {sheet}: {missing[:5]}...")
    if TEXT_COL not in df.columns:
        raise KeyError(f"Missing {TEXT_COL} on {sheet}")
    texts = df[TEXT_COL].fillna("").astype(str).tolist()
    labs = np.zeros((len(df), len(REGION_NAMES_29)), dtype=np.int64)
    for j, r in enumerate(REGION_NAMES_29):
        labs[:, j] = [normalize_label(v) for v in df[r].tolist()]
    return df, texts, labs


def label_stats(labs: np.ndarray) -> dict:
    flat = labs.reshape(-1)
    c = Counter(flat.tolist())
    total = len(flat)
    return {
        "blank": c.get(0, 0),
        "normal": c.get(1, 0),
        "abnormal": c.get(2, 0),
        "frac": {
            "blank": c.get(0, 0) / total,
            "normal": c.get(1, 0) / total,
            "abnormal": c.get(2, 0) / total,
        },
        "n_rows": labs.shape[0],
        "buckets": dict(Counter(sentence_bucket(labs[i]) for i in range(labs.shape[0]))),
    }


def mark_val_on_xlsx(xlsx: str, val_indices: np.ndarray, out_meta: Path):
    """Add is_val_BLUE_BERT_CLS on train_phase2_wodup (1=val); rewrite workbook via pandas."""
    print("Marking val column on xlsx (rewrite all sheets; may take ~10–30 min)...", flush=True)
    val_set = set(int(i) for i in val_indices)
    xls = pd.ExcelFile(xlsx, engine="openpyxl")
    sheet_frames = {}
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)
        if sheet == "train_phase2_wodup":
            flags = np.zeros(len(df), dtype=np.int64)
            for i in val_set:
                if 0 <= i < len(df):
                    flags[i] = 1
            df[VAL_COL] = flags
        sheet_frames[sheet] = df
    xls.close()
    tmp = str(xlsx) + ".tmp_with_val.xlsx"
    with pd.ExcelWriter(tmp, engine="openpyxl") as writer:
        for sheet, df in sheet_frames.items():
            df.to_excel(writer, sheet_name=sheet, index=False)
    os.replace(tmp, xlsx)
    meta = {
        "xlsx": xlsx,
        "sheet": "train_phase2_wodup",
        "val_col": VAL_COL,
        "n_val": int(len(val_indices)),
        "val_indices_0based": [int(i) for i in val_indices.tolist()],
    }
    out_meta.write_text(json.dumps(meta, indent=2))
    print(f"Saved val marks → {xlsx} col={VAL_COL}; meta {out_meta}", flush=True)


def tokenize_texts(texts, tokenizer, max_len: int):
    ids_list, lens = [], []
    for t in texts:
        enc = tokenizer.encode(
            t,
            add_special_tokens=True,
            max_length=max_len,
            truncation=True,
        )
        ids_list.append(torch.tensor(enc, dtype=torch.long))
        lens.append(len(enc))
    return ids_list, lens


class RegionSentDataset(Dataset):
    def __init__(self, token_ids, lengths, labels):
        self.token_ids = token_ids
        self.lengths = lengths
        self.labels = labels  # (N, 29)

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, i):
        return {
            "imp": self.token_ids[i],
            "len": self.lengths[i],
            "label": torch.tensor(self.labels[i], dtype=torch.long),
        }


def collate_fn(sample_list):
    tensor_list = [s["imp"] for s in sample_list]
    batched_imp = torch.nn.utils.rnn.pad_sequence(
        tensor_list, batch_first=True, padding_value=PAD_IDX
    )
    batched_label = torch.stack([s["label"] for s in sample_list], dim=0)  # (B, 29)
    len_list = [s["len"] for s in sample_list]
    return {"imp": batched_imp, "label": batched_label, "len": len_list}


def attention_masks(batch, src_len, device):
    bsz, maxlen = batch.size()
    mask = torch.zeros(bsz, maxlen, dtype=torch.float, device=device)
    for i, l in enumerate(src_len):
        mask[i, :l] = 1.0
    return mask


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    # collect preds/labels per region
    all_pred = [[] for _ in range(29)]
    all_gold = [[] for _ in range(29)]
    total_loss = 0.0
    n = 0
    loss_func = nn.CrossEntropyLoss(reduction="sum")
    for data in loader:
        batch = data["imp"].to(device)
        label = data["label"].to(device)  # (B, 29)
        src_len = data["len"]
        attn = attention_masks(batch, src_len, device)
        out = model(batch, attn)  # list 29 × (B, 3)
        bsz = batch.size(0)
        n += bsz
        for j in range(29):
            total_loss += loss_func(out[j], label[:, j]).item()
            pred = out[j].argmax(dim=-1).cpu().numpy()
            gold = label[:, j].cpu().numpy()
            all_pred[j].append(pred)
            all_gold[j].append(gold)
    model.train()
    metrics = {"loss": total_loss / max(n * 29, 1)}
    macro_f1s, micro_correct, micro_total = [], 0, 0
    per_class_f1 = {c: [] for c in range(3)}
    for j in range(29):
        p = np.concatenate(all_pred[j])
        g = np.concatenate(all_gold[j])
        micro_correct += (p == g).sum()
        micro_total += len(g)
        macro_f1s.append(f1_score(g, p, average="macro", labels=[0, 1, 2], zero_division=0))
        for c in range(3):
            per_class_f1[c].append(
                f1_score(g == c, p == c, zero_division=0) if (g == c).any() or (p == c).any() else 0.0
            )
    metrics["acc_micro"] = float(micro_correct / max(micro_total, 1))
    metrics["f1_macro_regions_mean"] = float(np.mean(macro_f1s))
    metrics["f1_blank_mean"] = float(np.mean(per_class_f1[0]))
    metrics["f1_normal_mean"] = float(np.mean(per_class_f1[1]))
    metrics["f1_abnormal_mean"] = float(np.mean(per_class_f1[2]))
    # primary selection metric: mean of normal+abnormal F1 (blank dominates)
    metrics["f1_nonblank_mean"] = 0.5 * (metrics["f1_normal_mean"] + metrics["f1_abnormal_mean"])
    return metrics


def prepare(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "cache"
    cache.mkdir(exist_ok=True)

    _, train_texts, train_labs = load_sheet_labels(args.xlsx, "train_phase1_wodup")
    phase2_df, phase2_texts, phase2_labs = load_sheet_labels(args.xlsx, "train_phase2_wodup")

    train_stat = label_stats(train_labs)
    phase2_stat = label_stats(phase2_labs)
    print("train_phase1_wodup stats:", json.dumps(train_stat, indent=2), flush=True)
    print("train_phase2_wodup stats:", json.dumps(phase2_stat, indent=2), flush=True)

    # stratified on phase2 to match phase2 bucket mix (≈ wodup train: abn-heavy)
    # also reweight allocation toward train bucket fractions
    train_buckets = train_stat["buckets"]
    n = phase2_labs.shape[0]
    rng = np.random.RandomState(args.seed)
    buckets = np.array([sentence_bucket(phase2_labs[i]) for i in range(n)])
    keys = ["all_blank", "nor_only", "abn_any"]
    # target counts from TRAIN proportions
    t_tot = sum(train_buckets.get(k, 0) for k in keys)
    target = {k: int(round(N_VAL * train_buckets.get(k, 0) / max(t_tot, 1))) for k in keys}
    # fix sum
    while sum(target.values()) < N_VAL:
        target["abn_any"] += 1
    while sum(target.values()) > N_VAL:
        for k in ["all_blank", "nor_only", "abn_any"]:
            if target[k] > 0 and sum(target.values()) > N_VAL:
                target[k] -= 1
    chosen = []
    for k in keys:
        idx = np.where(buckets == k)[0]
        take = min(target[k], len(idx))
        if take:
            chosen.append(rng.choice(idx, size=take, replace=False))
        target[k] -= take
    chosen = list(np.concatenate(chosen)) if chosen else []
    if len(chosen) < N_VAL:
        remain = np.setdiff1d(np.arange(n), np.array(chosen, dtype=int))
        # fill preferring train-like rare nor_only then abn
        order = []
        for k in ["nor_only", "abn_any", "all_blank"]:
            order.extend(remain[buckets[remain] == k].tolist())
        need = N_VAL - len(chosen)
        chosen.extend(order[:need])
    val_idx = np.array(sorted(chosen[:N_VAL]), dtype=int)

    val_labs = phase2_labs[val_idx]
    val_texts = [phase2_texts[i] for i in val_idx]
    val_stat = label_stats(val_labs)
    print("val(2700) stats:", json.dumps(val_stat, indent=2), flush=True)

    # flags csv + meta first (xlsx rewrite is slow; done after cache so train can start)
    phase2_df = phase2_df.copy()
    flags = np.zeros(len(phase2_df), dtype=np.int64)
    flags[val_idx] = 1
    phase2_df[VAL_COL] = flags
    phase2_df[[TEXT_COL, VAL_COL]].to_csv(out / "phase2_wodup_val_flags.csv", index=False)
    (out / "val_split_meta.json").write_text(
        json.dumps(
            {
                "xlsx": args.xlsx,
                "sheet": "train_phase2_wodup",
                "val_col": VAL_COL,
                "n_val": int(len(val_idx)),
                "val_indices_0based": [int(i) for i in val_idx.tolist()],
                "val_stats": val_stat,
                "train_stats": train_stat,
            },
            indent=2,
        )
    )

    print("Tokenizing...", flush=True)
    tokenizer = BertTokenizer.from_pretrained(args.pretrain_path)
    tr_ids, tr_lens = tokenize_texts(train_texts, tokenizer, args.max_len)
    va_ids, va_lens = tokenize_texts(val_texts, tokenizer, args.max_len)

    torch.save(
        {"ids": tr_ids, "lens": tr_lens, "labels": train_labs},
        cache / "train.pt",
    )
    torch.save(
        {"ids": va_ids, "lens": va_lens, "labels": val_labs, "val_indices": val_idx},
        cache / "val.pt",
    )
    (out / "prepare_stats.json").write_text(
        json.dumps(
            {
                "train": train_stat,
                "phase2": phase2_stat,
                "val": val_stat,
                "n_train": len(train_texts),
                "n_val": len(val_texts),
                "label_map": REGION_LABEL2ID,
                "regions": REGION_NAMES_29,
            },
            indent=2,
        )
    )
    if args.skip_xlsx_mark:
        print("Skip xlsx rewrite (--skip_xlsx_mark); flags in phase2_wodup_val_flags.csv", flush=True)
    else:
        print("Prepare cache done. Rewriting xlsx val column...", flush=True)
        mark_val_on_xlsx(args.xlsx, val_idx, out / "val_split_meta.json")
    print("Prepare done.", flush=True)


def train_loop(args):
    out_dir = Path(args.output_dir)
    cache = out_dir / "cache"
    train_pack = torch.load(cache / "train.pt", map_location="cpu", weights_only=False)
    val_pack = torch.load(cache / "val.pt", map_location="cpu", weights_only=False)

    train_ds = RegionSentDataset(train_pack["ids"], train_pack["lens"], train_pack["labels"])
    val_ds = RegionSentDataset(val_pack["ids"], val_pack["lens"], val_pack["labels"])
    train_ld = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_ld = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    model = bert_labeler(
        pretrain_path=args.pretrain_path,
        task="sentence_region",
        n_regions=29,
        n_classes=3,
        p=args.dropout,
    )
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss(reduction="sum")

    best_metric = -1.0
    begin = time.time()
    log_path = out_dir / "train_log.txt"
    results = []

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    log(
        f"start train n_train={len(train_ds)} n_val={len(val_ds)} "
        f"batch={args.batch_size} epochs={args.epochs} max_len={args.max_len}"
    )
    steps_per_epoch = len(train_ld)
    log(f"steps_per_epoch={steps_per_epoch}")

    global_step = 0
    for epoch in range(args.epochs):
        report_loss, report_n = 0.0, 0
        for i, data in enumerate(train_ld):
            batch = data["imp"].to(device)
            label = data["label"].to(device)  # (B, 29)
            src_len = data["len"]
            attn = attention_masks(batch, src_len, device)
            optimizer.zero_grad()
            logits = model(batch, attn)
            batch_loss = 0.0
            for j in range(29):
                batch_loss = batch_loss + loss_func(logits[j], label[:, j])
            bsz = batch.size(0)
            loss = batch_loss / bsz
            loss.backward()
            optimizer.step()
            report_loss += float(batch_loss.detach())
            report_n += bsz
            global_step += 1

            if (i + 1) % args.log_every == 0:
                elapsed = time.time() - begin
                avg = report_loss / max(report_n, 1)
                # ETA
                done_steps = epoch * steps_per_epoch + (i + 1)
                total_steps = args.epochs * steps_per_epoch
                eta = elapsed / done_steps * (total_steps - done_steps)
                log(
                    f"epoch {epoch+1}/{args.epochs} iter {i+1}/{steps_per_epoch} "
                    f"avg_loss {avg:.3f} elapsed {elapsed:.1f}s eta {eta/60:.1f}min"
                )
                report_loss, report_n = 0.0, 0

            if args.valid_niter > 0 and (i + 1) % args.valid_niter == 0:
                metrics = evaluate(model, val_ld, device)
                log(f"  mid-val: {json.dumps(metrics)}")
                results.append({"epoch": epoch + 1, "iter": i + 1, **metrics})
                if metrics["f1_nonblank_mean"] > best_metric:
                    best_metric = metrics["f1_nonblank_mean"]
                    path = out_dir / f"model_best_epoch{epoch+1}_iter{i+1}.pt"
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "iter": i + 1,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "metrics": metrics,
                            "label_map": REGION_LABEL2ID,
                            "regions": REGION_NAMES_29,
                            "pretrain_path": args.pretrain_path,
                            "task": "sentence_region",
                        },
                        path,
                    )
                    # stable name
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "iter": i + 1,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "metrics": metrics,
                            "label_map": REGION_LABEL2ID,
                            "regions": REGION_NAMES_29,
                            "pretrain_path": args.pretrain_path,
                            "task": "sentence_region",
                        },
                        out_dir / "model_best.pt",
                    )
                    log(f"  saved new best → {path} metric={best_metric:.4f}")

        # end-epoch val
        metrics = evaluate(model, val_ld, device)
        log(f"epoch {epoch+1} end-val: {json.dumps(metrics)}")
        results.append({"epoch": epoch + 1, "iter": "end", **metrics})
        ckpt = out_dir / f"model_epoch{epoch+1}.pt"
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
                "label_map": REGION_LABEL2ID,
                "regions": REGION_NAMES_29,
                "pretrain_path": args.pretrain_path,
                "task": "sentence_region",
            },
            ckpt,
        )
        if metrics["f1_nonblank_mean"] > best_metric:
            best_metric = metrics["f1_nonblank_mean"]
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": metrics,
                    "label_map": REGION_LABEL2ID,
                    "regions": REGION_NAMES_29,
                    "pretrain_path": args.pretrain_path,
                    "task": "sentence_region",
                },
                out_dir / "model_best.pt",
            )
            log(f"  saved new best model_best.pt metric={best_metric:.4f}")

    (out_dir / "val_results.json").write_text(json.dumps(results, indent=2))
    log(f"done. best_f1_nonblank_mean={best_metric:.4f} total_h={(time.time()-begin)/3600:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default=XLSX_DEFAULT)
    ap.add_argument("--pretrain_path", default=BLUE_BERT)
    ap.add_argument("--output_dir", default=OUT_DEFAULT)
    ap.add_argument("--prepare_only", action="store_true")
    ap.add_argument("--train_only", action="store_true")
    ap.add_argument("--mark_xlsx_only", action="store_true",
                    help="Only rewrite xlsx val column using val_split_meta.json")
    ap.add_argument("--skip_xlsx_mark", action="store_true",
                    help="During prepare, skip slow xlsx rewrite")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--valid_niter", type=int, default=2000)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.mark_xlsx_only:
        out = Path(args.output_dir)
        meta = json.loads((out / "val_split_meta.json").read_text())
        mark_val_on_xlsx(meta["xlsx"], np.array(meta["val_indices_0based"]), out / "val_split_meta.json")
        return

    if not args.train_only:
        prepare(args)
    if not args.prepare_only:
        train_loop(args)


if __name__ == "__main__":
    main()
