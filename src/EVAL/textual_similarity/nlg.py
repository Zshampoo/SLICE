"""NLG metrics: BLEU-1/4, ROUGE-L, METEOR, BertScore."""

from __future__ import annotations

import os
from typing import Dict, List, Sequence

import numpy as np


def _truncate_for_bert(
    texts: Sequence[str],
    model_type: str,
    max_length: int | None = None,
) -> List[str]:
    """Truncate to BERT position limit (PubMedBERT max positions = 512).

    bert_score uses tokenizer.model_max_length for truncation, but some
    tokenizers (e.g. PubMedBERT) expose an effectively unlimited value while
    the model still only supports 512 positions.
    """
    from bert_score.utils import get_tokenizer

    cap = int(os.environ.get("BERTSCORE_MAX_LENGTH", max_length or 512))
    tokenizer = get_tokenizer(model_type)
    out: List[str] = []
    for t in texts:
        s = str(t) if t is not None else ""
        ids = tokenizer.encode(
            s, add_special_tokens=True, max_length=cap, truncation=True
        )
        out.append(tokenizer.decode(ids, skip_special_tokens=True))
    return out


def _to_coco_dicts(refs: Sequence[str], hyps: Sequence[str]):
    gts, res = {}, {}
    for i, (r, h) in enumerate(zip(refs, hyps)):
        gts[i] = [str(r) if r is not None else ""]
        res[i] = [str(h) if h is not None else ""]
    return gts, res


def _meteor_mean(gts: dict, res: dict, batch_size: int = 200) -> float:
    """Corpus METEOR as mean of per-sample scores, computed in batches.

    pycocoevalcap Meteor feeds one giant EVAL line for all samples; with ~8k
    reports that deadlocks on the Java↔Python stdio pipes. Batching avoids it.
    """
    from pycocoevalcap.meteor.meteor import Meteor

    ids = list(gts.keys())
    all_scores: List[float] = []
    for start in range(0, len(ids), batch_size):
        chunk = ids[start : start + batch_size]
        def _flat(xs):
            return [str(x).replace("\r", " ").replace("\n", " ") for x in xs]

        g = {j: _flat(gts[i]) for j, i in enumerate(chunk)}
        r = {j: _flat(res[i]) for j, i in enumerate(chunk)}
        scorer = Meteor()
        try:
            _, scores = scorer.compute_score(g, r)
            all_scores.extend(float(s) for s in scores)
        finally:
            try:
                scorer.meteor_p.stdin.close()
                scorer.meteor_p.kill()
                scorer.meteor_p.wait(timeout=5)
            except Exception:
                pass
        print(f"[meteor] batch {start}-{start+len(chunk)-1}/{len(ids)}", flush=True)
    return float(np.mean(all_scores)) if all_scores else 0.0


def compute_nlg_coco(refs: Sequence[str], hyps: Sequence[str]) -> Dict[str, float]:
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge

    gts, res = _to_coco_dicts(refs, hyps)
    for k in res:
        res[k][0] = (res[k][0] + " ").replace(". ", " . ").replace(" - ", "-")

    out: Dict[str, float] = {}
    bleu, _ = Bleu(4).compute_score(gts, res)
    out["BL-1"] = float(bleu[0])
    out["BL-4"] = float(bleu[3])
    out["BLEU-4"] = float(bleu[3])
    out["BLEU_2"] = float(bleu[1])  # for RadCliQ

    out["MTR"] = _meteor_mean(gts, res, batch_size=200)

    rouge, _ = Rouge().compute_score(gts, res)
    out["RG-L"] = float(rouge)
    out["ROUGE-L"] = float(rouge)
    return out


def compute_nlg(refs: Sequence[str], hyps: Sequence[str]) -> Dict[str, float]:
    """Paper textual-similarity subset + common NLG aliases."""
    return compute_nlg_coco(refs, hyps)


def compute_bertscore(
    refs: Sequence[str],
    hyps: Sequence[str],
    model_type: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    batch_size: int = 32,
    device: str | None = None,
    rescale_with_baseline: bool = False,
    idf: bool = False,
    num_layers: int | None = 9,
) -> Dict[str, float]:
    from bert_score import score as bert_score_fn

    refs_l = [str(r) if r is not None else "" for r in refs]
    hyps_l = [str(h) if h is not None else "" for h in hyps]
    refs_l = _truncate_for_bert(refs_l, model_type)
    hyps_l = _truncate_for_bert(hyps_l, model_type)
    kwargs = dict(
        model_type=model_type,
        batch_size=batch_size,
        rescale_with_baseline=rescale_with_baseline,
        idf=idf,
        verbose=False,
    )
    if num_layers is not None:
        kwargs["num_layers"] = num_layers
    if device is not None:
        kwargs["device"] = device
    P, R, F1 = bert_score_fn(hyps_l, refs_l, lang="en", **kwargs)
    return {
        "BertScore": float(F1.mean().item()),
        "BertScore_P": float(P.mean().item()),
        "BertScore_R": float(R.mean().item()),
        "BertScore_per_sample": F1.cpu().numpy(),
    }


def compute_bleu2_per_sample(refs: Sequence[str], hyps: Sequence[str]) -> np.ndarray:
    """Per-sample BLEU-2 for RadCliQ (corpus-style scorer averaged per pair)."""
    from pycocoevalcap.bleu.bleu import Bleu

    scores = []
    scorer = Bleu(2)
    for r, h in zip(refs, hyps):
        gts = {0: [str(r) if r is not None else ""]}
        res = {0: [str(h) if h is not None else ""]}
        res[0][0] = (res[0][0] + " ").replace(". ", " . ").replace(" - ", "-")
        try:
            bleu, _ = scorer.compute_score(gts, res)
        except Exception:
            bleu = [0.0, 0.0]
        scores.append(float(bleu[1]))
    return np.asarray(scores, dtype=np.float64)
