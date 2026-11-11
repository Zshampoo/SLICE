#!/usr/bin/env python3
"""Train report→29-region {normal=0, abnormal=1} binary classifier (BLUE_BERT).

Train: train_phase1 sheet of report_region_label.xlsx (reference_report + 29 regions).
Val: stratified 1000 from train_phase2 (class mix matched to train); marks is_val column.
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
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer

RGRG_ROOT = Path(__file__).resolve().parents[3]
if str(RGRG_ROOT) not in sys.path:
    sys.path.insert(0, str(RGRG_ROOT))

from src.CheXbert.src.models.bert_labeler import (  # noqa: E402
    REGION_NAMES_29,
    REPORT_REGION_LABEL2ID,
    bert_labeler,
)

_DATA = Path(__import__("os").environ.get("DATA_ROOT", "./data"))
XLSX_DEFAULT = str(_DATA / "BLUE_BERT_CLS_REPORT" / "report_region_label.xlsx")
BLUE_BERT = str(_DATA / "weights" / "bluebert")
OUT_DEFAULT = str(_DATA / "runs" / "bluebert_report_region")
VAL_COL = "is_val_BLUE_BERT_CLS_REPORT"
TEXT_COL = "reference_report"
N_VAL = 1000


def normalize_label(v) -> int:
    """Map Normal→0, Abnormal→1."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        raise ValueError("Blank/NaN region label not allowed for report_region task")
    s = str(v).strip().lower()
    if s not in REPORT_REGION_LABEL2ID:
        raise ValueError(f"Unknown label: {v!r} (expect normal/abnormal)")
    return REPORT_REGION_LABEL2ID[s]


def sample_bucket(lab_row: np.ndarray) -> str:
    """Stratify by presence of abnormal (matches train mix ~74% abn_any)."""
    return "abn_any" if (lab_row == 1).any() else "nor_only"


def load_sheet(xlsx: str, sheet: str):
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
    n0 = int((flat == 0).sum())
    n1 = int((flat == 1).sum())
    total = max(len(flat), 1)
    buckets = Counter(sample_bucket(labs[i]) for i in range(len(labs)))
    return {
        "normal": n0,
        "abnormal": n1,
        "frac": {"normal": n0 / total, "abnormal": n1 / total},
        "n_rows": int(labs.shape[0]),
        "buckets": dict(buckets),
        "abn_rate_per_region": {
            REGION_NAMES_29[j]: float(labs[:, j].mean()) for j in range(29)
        },
    }


def stratified_val_indices(labs: np.ndarray, n_val: int, seed: int = 42) -> np.ndarray:
    """Draw n_val indices matching train bucket proportions (abn_any / nor_only)."""
    rng = np.random.RandomState(seed)
    buckets = {k: [] for k in ("abn_any", "nor_only")}
    for i in range(len(labs)):
        buckets[sample_bucket(labs[i])].append(i)
    # target proportions from this set itself (phase2) but user asked match train —
    # caller should pass train proportions via reweight; here we use phase2 mix first,
    # then adjust counts to train_frac if provided through labs_train optional path.
    total = len(labs)
    frac_abn = len(buckets["abn_any"]) / max(total, 1)
    n_abn = int(round(n_val * frac_abn))
    n_nor = n_val - n_abn
    # clamp
    n_abn = min(n_abn, len(buckets["abn_any"]))
    n_nor = min(n_nor, len(buckets["nor_only"]))
    # fill remainder from the larger pool
    chosen = []
    chosen.extend(rng.choice(buckets["abn_any"], size=n_abn, replace=False).tolist())
    chosen.extend(rng.choice(buckets["nor_only"], size=n_nor, replace=False).tolist())
    rem = n_val - len(chosen)
    if rem > 0:
        pool = [i for i in range(total) if i not in set(chosen)]
        chosen.extend(rng.choice(pool, size=rem, replace=False).tolist())
    return np.array(sorted(chosen), dtype=np.int64)


def stratified_val_indices_match_train(
    labs_phase2: np.ndarray, labs_train: np.ndarray, n_val: int, seed: int = 42
) -> np.ndarray:
    """Match train abn_any fraction when sampling val from phase2."""
    rng = np.random.RandomState(seed)
    buckets = {k: [] for k in ("abn_any", "nor_only")}
    for i in range(len(labs_phase2)):
        buckets[sample_bucket(labs_phase2[i])].append(i)
    train_abn_frac = sum(sample_bucket(labs_train[i]) == "abn_any" for i in range(len(labs_train))) / max(
        len(labs_train), 1
    )
    n_abn = int(round(n_val * train_abn_frac))
    n_nor = n_val - n_abn
    n_abn = min(max(n_abn, 0), len(buckets["abn_any"]))
    n_nor = min(max(n_nor, 0), len(buckets["nor_only"]))
    chosen = []
    if n_abn:
        chosen.extend(rng.choice(buckets["abn_any"], size=n_abn, replace=False).tolist())
    if n_nor:
        chosen.extend(rng.choice(buckets["nor_only"], size=n_nor, replace=False).tolist())
    rem = n_val - len(chosen)
    if rem > 0:
        pool = [i for i in range(len(labs_phase2)) if i not in set(chosen)]
        chosen.extend(rng.choice(pool, size=min(rem, len(pool)), replace=False).tolist())
    return np.array(sorted(chosen), dtype=np.int64)


def mark_val_on_xlsx(xlsx: str, val_idx: np.ndarray, meta_path: Path):
    """Add/overwrite is_val_BLUE_BERT_CLS_REPORT on train_phase2 sheet."""
    print(f"Marking {VAL_COL} on train_phase2 ...", flush=True)
    df1 = pd.read_excel(xlsx, sheet_name="train_phase1", engine="openpyxl")
    df2 = pd.read_excel(xlsx, sheet_name="train_phase2", engine="openpyxl")
    flags = np.zeros(len(df2), dtype=np.int64)
    flags[val_idx] = 1
    df2[VAL_COL] = flags
    # also ensure phase1 has column = 0 for consistency
    if VAL_COL not in df1.columns:
        df1[VAL_COL] = 0
    else:
        df1[VAL_COL] = 0
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        df1.to_excel(writer, sheet_name="train_phase1", index=False)
        df2.to_excel(writer, sheet_name="train_phase2", index=False)
    meta = {
        "val_col": VAL_COL,
        "n_val": int(flags.sum()),
        "n_phase2": int(len(df2)),
        "val_indices": val_idx.tolist(),
        "val_dicom_ids": df2.loc[flags == 1, "dicom_id"].astype(str).tolist()
        if "dicom_id" in df2.columns
        else [],
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {VAL_COL}: {int(flags.sum())}/{len(df2)} marked for val", flush=True)


def tokenize_texts(texts, tokenizer, max_len: int):
    enc = tokenizer(
        texts,
        padding=False,
        truncation=True,
        max_length=max_len,
        return_attention_mask=False,
        add_special_tokens=True,
    )
    ids = [torch.tensor(x, dtype=torch.long) for x in enc["input_ids"]]
    lens = [len(x) for x in ids]
    return ids, lens


class RegionReportDataset(Dataset):
    def __init__(self, ids, lens, labels):
        self.ids, self.lens, self.labels = ids, lens, labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return {"imp": self.ids[i], "len": int(self.lens[i]), "label": self.labels[i]}


def collate_fn(batch):
    max_len = max(x["len"] for x in batch)
    B = len(batch)
    imp = torch.zeros(B, max_len, dtype=torch.long)
    label = torch.stack([torch.as_tensor(x["label"], dtype=torch.long) for x in batch], 0)
    lens = [x["len"] for x in batch]
    for i, x in enumerate(batch):
        L = x["len"]
        t = x["imp"] if torch.is_tensor(x["imp"]) else torch.tensor(x["imp"])
        imp[i, :L] = t[:L]
    return {"imp": imp, "label": label, "len": lens}


def attention_masks(batch, src_len, device):
    B, T = batch.size()
    mask = torch.zeros(B, T, device=device)
    for i, L in enumerate(src_len):
        mask[i, : int(L)] = 1
    return mask


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_pred = [[] for _ in range(29)]
    all_gold = [[] for _ in range(29)]
    all_prob1 = [[] for _ in range(29)]  # P(abnormal)
    total_loss, n = 0.0, 0
    loss_func = nn.CrossEntropyLoss(reduction="sum")
    for data in loader:
        batch = data["imp"].to(device)
        label = data["label"].to(device)
        attn = attention_masks(batch, data["len"], device)
        out = model(batch, attn)  # 29 × (B, 2)
        bsz = batch.size(0)
        n += bsz
        for j in range(29):
            total_loss += loss_func(out[j], label[:, j]).item()
            prob = torch.softmax(out[j], dim=-1)
            pred = prob.argmax(dim=-1).cpu().numpy()
            gold = label[:, j].cpu().numpy()
            all_pred[j].append(pred)
            all_gold[j].append(gold)
            all_prob1[j].append(prob[:, 1].cpu().numpy())
    model.train()

    metrics = {"loss": total_loss / max(n * 29, 1)}
    macro_f1s, micro_correct, micro_total = [], 0, 0
    f1_nor, f1_abn, aucs = [], [], []
    for j in range(29):
        p = np.concatenate(all_pred[j])
        g = np.concatenate(all_gold[j])
        pr = np.concatenate(all_prob1[j])
        micro_correct += (p == g).sum()
        micro_total += len(g)
        macro_f1s.append(f1_score(g, p, average="macro", labels=[0, 1], zero_division=0))
        f1_nor.append(f1_score(g == 0, p == 0, zero_division=0) if ((g == 0).any() or (p == 0).any()) else 0.0)
        f1_abn.append(f1_score(g == 1, p == 1, zero_division=0) if ((g == 1).any() or (p == 1).any()) else 0.0)
        try:
            if len(np.unique(g)) >= 2:
                aucs.append(float(roc_auc_score(g, pr)))
        except Exception:
            pass
    metrics["acc_micro"] = float(micro_correct / max(micro_total, 1))
    metrics["f1_macro_regions_mean"] = float(np.mean(macro_f1s))
    metrics["f1_normal_mean"] = float(np.mean(f1_nor))
    metrics["f1_abnormal_mean"] = float(np.mean(f1_abn))
    # primary: mean of normal+abnormal F1 (balanced)
    metrics["f1_binary_mean"] = 0.5 * (metrics["f1_normal_mean"] + metrics["f1_abnormal_mean"])
    metrics["auc_mean"] = float(np.mean(aucs)) if aucs else float("nan")
    return metrics


def prepare(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "cache"
    cache.mkdir(exist_ok=True)

    df_tr, train_texts, train_labs = load_sheet(args.xlsx, "train_phase1")
    df_p2, phase2_texts, phase2_labs = load_sheet(args.xlsx, "train_phase2")
    st_tr = label_stats(train_labs)
    st_p2 = label_stats(phase2_labs)
    print("train_phase1 stats:", json.dumps(st_tr, indent=2)[:800], flush=True)
    print("train_phase2 stats:", json.dumps(st_p2, indent=2)[:800], flush=True)

    val_idx = stratified_val_indices_match_train(phase2_labs, train_labs, args.n_val, seed=args.seed)
    val_texts = [phase2_texts[i] for i in val_idx]
    val_labs = phase2_labs[val_idx]
    st_val = label_stats(val_labs)
    print("val(1000) stats:", json.dumps(st_val, indent=2)[:800], flush=True)

    prepare_stats = {
        "train": st_tr,
        "phase2": st_p2,
        "val": st_val,
        "n_train": len(train_labs),
        "n_val": len(val_labs),
        "label_map": REPORT_REGION_LABEL2ID,
        "regions": REGION_NAMES_29,
        "text_col": TEXT_COL,
        "val_col": VAL_COL,
    }
    (out / "prepare_stats.json").write_text(json.dumps(prepare_stats, indent=2))

    tokenizer = BertTokenizer.from_pretrained(args.pretrain_path)
    print("Tokenizing...", flush=True)
    tr_ids, tr_lens = tokenize_texts(train_texts, tokenizer, args.max_len)
    va_ids, va_lens = tokenize_texts(val_texts, tokenizer, args.max_len)
    torch.save(
        {"ids": tr_ids, "lens": tr_lens, "labels": torch.tensor(train_labs, dtype=torch.long)},
        cache / "train.pt",
    )
    torch.save(
        {"ids": va_ids, "lens": va_lens, "labels": torch.tensor(val_labs, dtype=torch.long)},
        cache / "val.pt",
    )
    print("Prepare cache done. Rewriting xlsx val column...", flush=True)
    if not args.skip_xlsx_mark:
        mark_val_on_xlsx(args.xlsx, val_idx, out / "val_split_meta.json")
    else:
        (out / "val_split_meta.json").write_text(
            json.dumps({"val_indices": val_idx.tolist(), "n_val": len(val_idx)}, indent=2)
        )
    print("Prepare done.", flush=True)


def train_loop(args):
    out_dir = Path(args.output_dir)
    cache = out_dir / "cache"
    train_pack = torch.load(cache / "train.pt", map_location="cpu", weights_only=False)
    val_pack = torch.load(cache / "val.pt", map_location="cpu", weights_only=False)

    train_ds = RegionReportDataset(train_pack["ids"], train_pack["lens"], train_pack["labels"])
    val_ds = RegionReportDataset(val_pack["ids"], val_pack["lens"], val_pack["labels"])
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
        task="report_region",
        n_regions=29,
        n_classes=2,
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
        f"batch={args.batch_size} epochs={args.epochs} max_len={args.max_len} "
        f"task=report_region n_classes=2 label_map={REPORT_REGION_LABEL2ID}"
    )
    steps_per_epoch = len(train_ld)
    log(f"steps_per_epoch={steps_per_epoch}")
    # ETA note
    log(
        f"ETA_hint: ~{steps_per_epoch * args.epochs} steps; "
        f"sentence_ref was ~1.35h for 67k steps@max_len128 → "
        f"expect ~{(steps_per_epoch * args.epochs / 67520) * 1.35 * (args.max_len / 128):.1f}h"
    )

    for epoch in range(args.epochs):
        report_loss, report_n = 0.0, 0
        for i, data in enumerate(train_ld):
            batch = data["imp"].to(device)
            label = data["label"].to(device)
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

            if (i + 1) % args.log_every == 0:
                elapsed = time.time() - begin
                avg = report_loss / max(report_n, 1)
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
                if metrics["f1_binary_mean"] > best_metric:
                    best_metric = metrics["f1_binary_mean"]
                    payload = {
                        "epoch": epoch + 1,
                        "iter": i + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "metrics": metrics,
                        "label_map": REPORT_REGION_LABEL2ID,
                        "regions": REGION_NAMES_29,
                        "pretrain_path": args.pretrain_path,
                        "task": "report_region",
                    }
                    path = out_dir / f"model_best_epoch{epoch+1}_iter{i+1}.pt"
                    torch.save(payload, path)
                    torch.save(payload, out_dir / "model_best.pt")
                    log(f"  saved new best → {path} metric={best_metric:.4f}")

        metrics = evaluate(model, val_ld, device)
        log(f"epoch {epoch+1} end-val: {json.dumps(metrics)}")
        results.append({"epoch": epoch + 1, "iter": "end", **metrics})
        ckpt = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "label_map": REPORT_REGION_LABEL2ID,
            "regions": REGION_NAMES_29,
            "pretrain_path": args.pretrain_path,
            "task": "report_region",
        }
        torch.save(ckpt, out_dir / f"model_epoch{epoch+1}.pt")
        if metrics["f1_binary_mean"] > best_metric:
            best_metric = metrics["f1_binary_mean"]
            torch.save(ckpt, out_dir / "model_best.pt")
            log(f"  saved new best model_best.pt metric={best_metric:.4f}")

    (out_dir / "val_results.json").write_text(json.dumps(results, indent=2))
    log(f"done. best_f1_binary_mean={best_metric:.4f} total_h={(time.time()-begin)/3600:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default=XLSX_DEFAULT)
    ap.add_argument("--pretrain_path", default=BLUE_BERT)
    ap.add_argument("--output_dir", default=OUT_DEFAULT)
    ap.add_argument("--prepare_only", action="store_true")
    ap.add_argument("--train_only", action="store_true")
    ap.add_argument("--skip_xlsx_mark", action="store_true")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--valid_niter", type=int, default=1000)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n_val", type=int, default=N_VAL)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.train_only:
        train_loop(args)
        return
    prepare(args)
    if args.prepare_only:
        return
    train_loop(args)


if __name__ == "__main__":
    main()
