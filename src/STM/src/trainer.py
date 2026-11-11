from pathlib import Path
# src/trainer.py
import json
import torch
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)

from src.LLM.CE_calc import clean_report_mimic_cxr, compute_clinical_efficacy_scores, create_language_model_scores_dict, \
    compute_report_level_scores
from src.STM.src.config import Config
import numpy as np
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from transformers.utils import logging
import sys
import os
import contextlib


@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
            yield


def _format_metrics_detail(metrics: dict, step: int = None, epoch: float = None) -> str:
    # HuggingFace Trainer prefixes eval metrics with "eval_"
    norm = {}
    for k, v in metrics.items():
        nk = k[5:] if k.startswith("eval_") else k
        norm[nk] = v

    lines = []
    lines.append("==== Validation metrics detail ====")
    if step is not None:
        lines.append(f"global_step: {step}")
    if epoch is not None:
        lines.append(f"epoch: {epoch}")
    if "loss" in norm:
        lines.append(f"eval_loss: {norm['loss']}")
    lines.append("")
    lines.append("-- NLG --")
    for k in ["bleu1", "bleu2", "bleu3", "bleu4", "meteor", "rougeL", "cider"]:
        if k in norm:
            lines.append(f"{k}: {norm[k]}")
    lines.append("")
    lines.append("-- Clinical Efficacy --")
    for k in ["F1_micro_all", "F1_micro_conflict6", "CE_P", "CE_R", "CE_f1_5"]:
        if k in norm:
            lines.append(f"{k}: {norm[k]}")
    lines.append("")
    lines.append("-- RadGraph --")
    for k in ["radgraph_partial", "radgraph_complete", "radgraph_simple", "radgraph_f1"]:
        if k in norm:
            lines.append(f"{k}: {norm[k]}")
    lines.append("")
    lines.append("-- RateScore --")
    for k in ["RateScore"]:
        if k in norm:
            lines.append(f"{k}: {norm[k]}")
    lines.append("")
    lines.append("-- All returned metrics --")
    for k, v in sorted(metrics.items()):
        lines.append(f"{k}: {v}")
    lines.append("")
    return "\n".join(lines)


class PrintTrainLossCallback(TrainerCallback):
    """Print train / eval loss clearly; append to train_loss_live.log & eval_loss_live.log."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        ep = state.epoch
        step = state.global_step
        # train loss (HF puts training loss under "loss" without eval_ prefix)
        if "loss" in logs and "eval_loss" not in logs:
            loss = logs.get("loss")
            lr = logs.get("learning_rate")
            gn = logs.get("grad_norm")
            line = (
                f"[train_loss] step={step} epoch={ep:.4f} "
                f"loss={loss:.4f} grad_norm={gn} lr={lr}"
            )
            print(line, flush=True)
            try:
                row = (
                    f"[train_loss] step={step} epoch={float(ep):.4f} "
                    f"loss={float(loss):.4f} "
                    f"grad_norm={float(gn) if gn is not None else float('nan'):.4f} "
                    f"lr={float(lr) if lr is not None else float('nan'):.6e}\n"
                )
                with open(os.path.join(args.output_dir, "train_loss_live.log"), "a", encoding="utf-8") as f:
                    f.write(row)
                with open(os.path.join(args.output_dir, "loss_train_eval_live.log"), "a", encoding="utf-8") as f:
                    f.write(row)
            except Exception:
                pass
        # eval loss may also appear in on_log
        if "eval_loss" in logs:
            el = logs["eval_loss"]
            line = f"[eval_loss] step={step} epoch={ep:.4f} loss={float(el):.4f}"
            print(line, flush=True)
            try:
                with open(os.path.join(args.output_dir, "eval_loss_live.log"), "a", encoding="utf-8") as f:
                    f.write(line + "\n")
                with open(os.path.join(args.output_dir, "loss_train_eval_live.log"), "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                pass

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        el = metrics.get("eval_loss", metrics.get("loss"))
        if el is None:
            return
        ep = state.epoch
        step = state.global_step
        line = f"[eval_loss] step={step} epoch={ep:.4f} loss={float(el):.4f}"
        print(line, flush=True)
        try:
            path = os.path.join(args.output_dir, "eval_loss_live.log")
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            # also mirror into consolidated loss log
            both = os.path.join(args.output_dir, "loss_train_eval_live.log")
            with open(both, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass


class SaveEvalMetricsCallback(TrainerCallback):
    """Write detailed validation metrics into output_dir and each checkpoint folder.

    Also APPEND every eval into a single shared log file (eval_metrics_all.log)
    with step / epoch clearly marked.
    """

    def __init__(self):
        self.last_metrics = None
        self.last_step = None
        self.last_epoch = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        self.last_metrics = dict(metrics)
        self.last_step = state.global_step
        self.last_epoch = state.epoch
        detail = _format_metrics_detail(metrics, step=state.global_step, epoch=state.epoch)

        os.makedirs(args.output_dir, exist_ok=True)
        step_path = os.path.join(args.output_dir, f"eval_metrics_step_{state.global_step}.txt")
        with open(step_path, "w", encoding="utf-8") as f:
            f.write(detail)
        latest_path = os.path.join(args.output_dir, "eval_metrics_latest.txt")
        with open(latest_path, "w", encoding="utf-8") as f:
            f.write(detail)
        json_path = os.path.join(args.output_dir, f"eval_metrics_step_{state.global_step}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        # Append to consolidated all-eval log
        all_name = getattr(Config, "eval_all_log_name", "eval_metrics_all.log")
        all_path = os.path.join(args.output_dir, all_name)
        header = (
            f"\n{'=' * 72}\n"
            f"EVAL STEP={state.global_step}  epoch={state.epoch}\n"
            f"{'=' * 72}\n"
        )
        with open(all_path, "a", encoding="utf-8") as f:
            f.write(header)
            f.write(detail)
            # one-line summary for quick scanning
            norm = { (k[5:] if k.startswith("eval_") else k): v for k, v in metrics.items() }
            f.write(
                "[summary] "
                f"step={state.global_step} "
                f"eval_loss={norm.get('loss')} "
                f"BLEU-1={norm.get('bleu1')} BLEU-4={norm.get('bleu4')} "
                f"ROUGE={norm.get('rougeL')} METEOR={norm.get('meteor')} "
                f"P-all={norm.get('CE_P')} R-all={norm.get('CE_R')} "
                f"F1-all={norm.get('F1_micro_all')} F1_c6={norm.get('F1_micro_conflict6')} "
                f"RG-p={norm.get('radgraph_partial')} RG-c={norm.get('radgraph_complete')} "
                f"RateScore={norm.get('RateScore')}\n"
            )

        ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.isdir(ckpt):
            with open(os.path.join(ckpt, "eval_metrics_detail.txt"), "w", encoding="utf-8") as f:
                f.write(detail)
            with open(os.path.join(ckpt, "eval_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)

    def on_save(self, args, state, control, **kwargs):
        if not self.last_metrics:
            return
        ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(ckpt):
            return
        # Prefer metrics from the same step; otherwise attach latest eval.
        detail = _format_metrics_detail(
            self.last_metrics,
            step=self.last_step,
            epoch=self.last_epoch,
        )
        extra = ""
        if self.last_step != state.global_step:
            extra = (
                f"\n(note: metrics above are from eval step {self.last_step}, "
                f"while this checkpoint is step {state.global_step})\n"
            )
        with open(os.path.join(ckpt, "eval_metrics_detail.txt"), "w", encoding="utf-8") as f:
            f.write(detail + extra)
        with open(os.path.join(ckpt, "eval_metrics.json"), "w", encoding="utf-8") as f:
            payload = dict(self.last_metrics)
            payload["_checkpoint_step"] = state.global_step
            payload["_metrics_eval_step"] = self.last_step
            json.dump(payload, f, indent=2, ensure_ascii=False)


def build_compute_metrics(tokenizer):
    def safe_batch_decode(tokenizer, sequences):
        clean_sequences = []
        for i, seq in enumerate(sequences):
            if hasattr(seq, "tolist"):
                seq = seq.tolist()
            clean_seq = [
                int(token) for token in seq
                if isinstance(token, (int, float)) and token is not None and token != -100
            ]
            clean_sequences.append(clean_seq)
        return tokenizer.batch_decode(clean_sequences, skip_special_tokens=True)

    # Lazy RadGraph scorer (reuse across eval steps); cuda=0 under CUDA_VISIBLE_DEVICES
    _radgraph_scorer = {"obj": None}

    def compute_metrics(eval_preds):
        predictions, labels = eval_preds
        print(f"[eval] decoding predictions n={len(predictions)} ...", flush=True)
        decoded_preds = safe_batch_decode(tokenizer, predictions)
        decoded_labels = safe_batch_decode(tokenizer, labels)

        gts_list = list(decoded_labels)
        res_list = list(decoded_preds)

        language_model_scores = create_language_model_scores_dict()
        ref_reports_clean = [clean_report_mimic_cxr(x) for x in gts_list]
        pred_reports_clean = [clean_report_mimic_cxr(x) for x in res_list]
        print("[eval] computing NLG ...", flush=True)
        with suppress_output():
            compute_report_level_scores(language_model_scores, pred_reports_clean, ref_reports_clean)
        print("[eval] computing CE (CheXbert) ...", flush=True)
        with suppress_output():
            compute_clinical_efficacy_scores(language_model_scores, pred_reports_clean, ref_reports_clean)

        print("[eval] computing RadGraph (partial=-p, complete=-c) ...", flush=True)
        try:
            from src.LLM.LLM_test_cal import calc_radGraghF1
            if _radgraph_scorer["obj"] is None:
                from radgraph import F1RadGraph
                _radgraph_scorer["obj"] = F1RadGraph(
                    reward_level="all", model_type="radgraph-xl", cuda=0
                )
            rg = calc_radGraghF1(
                pred_reports_clean,
                ref_reports_clean,
                is_clean=False,  # already cleaned
                cuda=0,
                scorer=_radgraph_scorer["obj"],
            )
        except Exception as e:
            print(f"[eval] RadGraph failed: {e}", flush=True)
            rg = {
                "radgraph_partial": float("nan"),
                "radgraph_complete": float("nan"),
                "radgraph_simple": float("nan"),
                "radgraph_f1": float("nan"),
            }

        print("[eval] computing RateScore ...", flush=True)
        rate = float("nan")
        try:
            rg_eval_root = str(Path(os.environ.get("DATA_ROOT", "./data")) / "metrics")
            mlrg_site = os.environ.get("MLRG_SITE", "")
            for p in (rg_eval_root, mlrg_site):
                if p not in sys.path:
                    sys.path.insert(0, p)
            from rg_eval.ratescore_metric import compute_ratescore

            rs = compute_ratescore(ref_reports_clean, pred_reports_clean, use_gpu=True)
            rate = float(rs.get("RateScore", float("nan")))
        except Exception as e:
            print(f"[eval] RateScore failed: {e}", flush=True)

        report = language_model_scores["report"]
        ce = report["CE"]
        metrics = {
            "bleu1": report["bleu_1"],
            "bleu2": report["bleu_2"],
            "bleu3": report["bleu_3"],
            "bleu4": report["bleu_4"],
            "meteor": report["meteor"],
            "rougeL": report["rouge"],
            "cider": report["cider"],
            "F1_micro_all": ce["f1_micro_all"],
            "F1_micro_conflict6": ce["f1_micro_conflict_6"],
            "CE_P": ce["precision_micro_all"],
            "CE_R": ce["recall_micro_all"],
            "CE_f1_5": ce["f1_micro_5"],
            "radgraph_partial": rg.get("radgraph_partial"),
            "radgraph_complete": rg.get("radgraph_complete"),
            "radgraph_simple": rg.get("radgraph_simple"),
            "radgraph_f1": rg.get("radgraph_f1"),
            "RateScore": rate,
        }
        print(
            f"[eval] done NLG bleu4={metrics['bleu4']:.4f} cider={metrics['cider']:.4f} "
            f"F1_all={metrics['F1_micro_all']:.4f} F1_c6={metrics['F1_micro_conflict6']:.4f} "
            f"RG-p={metrics['radgraph_partial']} RG-c={metrics['radgraph_complete']} "
            f"RateScore={metrics['RateScore']}",
            flush=True,
        )
        return metrics

    return compute_metrics

def build_trainer(model, tokenizer, train_ds, test_ds, output_dir):
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="no",  # 这里没有验证集，所以不需要评估
        learning_rate=Config.learning_rate,
        per_device_train_batch_size=Config.train_batch_size,
        per_device_eval_batch_size=Config.eval_batch_size,
        num_train_epochs=Config.num_train_epochs,
        # save_total_limit=3,
        predict_with_generate=True,
        logging_dir=output_dir,
    )
    # data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    data_collator = CustomDataCollator(tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator = data_collator
    )


    return trainer

def build_trainer_weval(model, tokenizer, train_ds, test_ds, output_dir,
                        eval_steps=300, save_steps=300, max_steps=-1,
                        num_train_epochs=None, generation_max_length=None,
                        save_total_limit=None):
    """
    save_total_limit=None: keep EVERY checkpoint-{{step}} (300/600/.../end).
    Do not set a small limit — HF Trainer would delete older folders.
    """
    # Match eval_to_ts.predict_batch_wprompt. Do not also set generation_max_length:
    # that kwarg becomes max_length and overrides max_new_tokens=150.
    gen_config = model.generation_config
    gen_config.max_new_tokens = 150
    gen_config.num_beams = 4
    gen_config.length_penalty = 1.0
    gen_config.repetition_penalty = 1.0
    gen_config.eos_token_id = tokenizer.sep_token_id
    gen_config.pad_token_id = tokenizer.pad_token_id
    if generation_max_length is not None:
        gen_config.max_new_tokens = generation_max_length
    print(
        "[gen] eval_to_ts settings: num_beams=4 length_penalty=1.0 "
        f"repetition_penalty=1.0 max_new_tokens={gen_config.max_new_tokens} "
        f"eos={gen_config.eos_token_id} pad={gen_config.pad_token_id}",
        flush=True,
    )
    training_args_kwargs = dict(
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        learning_rate=Config.learning_rate,
        per_device_train_batch_size=Config.train_batch_size,
        per_device_eval_batch_size=Config.eval_batch_size,
        num_train_epochs=Config.num_train_epochs if num_train_epochs is None else num_train_epochs,
        max_steps=max_steps,
        predict_with_generate=True,
        generation_num_beams=4,
        generation_config=gen_config,
        logging_strategy="steps",
        logging_steps=max(1, min(50, eval_steps)),
        logging_dir=os.path.join(output_dir, "logs"),
        load_best_model_at_end=True,
        metric_for_best_model="F1_micro_all",
        greater_is_better=True,
        report_to=[],
    )
    # Omit save_total_limit when None so HF keeps all checkpoint-{step} folders
    if save_total_limit is not None:
        training_args_kwargs["save_total_limit"] = save_total_limit
    # Prefer saving model weights only if supported (much less disk for ~40 ckpts)
    try:
        import inspect
        if "save_only_model" in inspect.signature(Seq2SeqTrainingArguments.__init__).parameters:
            training_args_kwargs["save_only_model"] = False  # keep full ckpt (optimizer) for resume
    except Exception:
        pass
    training_args = Seq2SeqTrainingArguments(**training_args_kwargs)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    compute_metrics = build_compute_metrics(tokenizer)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[SaveEvalMetricsCallback(), PrintTrainLossCallback()],
    )

    return trainer

from transformers import BartTokenizer

class CustomDataCollator(DataCollatorForSeq2Seq):
    def __init__(self, tokenizer: BartTokenizer, model=None):
        super().__init__(tokenizer, model)
        self.tokenizer = tokenizer

    def shift_right(self, labels: torch.Tensor) -> torch.Tensor:
        pad_token_id = self.tokenizer.pad_token_id
        bos_token_id = self.tokenizer.bos_token_id

        safe_labels = torch.where(labels == -100, torch.full_like(labels, pad_token_id), labels)
        shifted = safe_labels.new_full(safe_labels.shape, pad_token_id)
        shifted[:, 1:] = safe_labels[:, :-1]
        shifted[:, 0] = bos_token_id
        return shifted

    def __call__(self, features):
        input_ids = [torch.tensor(f["input_ids"]) for f in features]
        labels = [torch.tensor(f["labels"]) for f in features]
        shifted_decoder_input_ids = [torch.tensor(f["decoder_input_ids"]) for f in features]

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True,
                                                    padding_value=self.tokenizer.pad_token_id)

        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True,
                                                 padding_value=-100)

        shifted_decoder_input_ids = torch.nn.utils.rnn.pad_sequence(shifted_decoder_input_ids, batch_first=True,
                                                            padding_value=self.tokenizer.pad_token_id)

        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        decoder_attention_mask = (shifted_decoder_input_ids != self.tokenizer.pad_token_id).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "decoder_input_ids": shifted_decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
        }
