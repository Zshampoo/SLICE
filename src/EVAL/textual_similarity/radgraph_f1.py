"""LLM / NLG / RadGraph evaluation helpers."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence, Union

import pandas as pd
from radgraph import F1RadGraph

from src.EVAL.factual_accuracy.CE_calc import clean_report_mimic_cxr

ReportSeq = Sequence[Union[str, float, int, None]]


def _to_text_list(reports: ReportSeq, is_clean: bool) -> list[str]:
    out: list[str] = []
    for x in reports:
        if pd.isna(x):
            out.append("")
            continue
        s = str(x)
        if is_clean:
            s = clean_report_mimic_cxr(s)
        out.append(s)
    return out


def calc_radGraghF1(
    hyps: ReportSeq,
    refs: ReportSeq,
    is_clean: bool = True,
    cuda: int = 0,
    model_type: str = "radgraph-xl",
    reward_level: str = "all",
    scorer: F1RadGraph | None = None,
) -> Mapping[str, float]:
    """
    Compute RadGraph F1 between hypothesis and reference report lists.

    Returns simple / partial / complete mean F1 (reward_level="all").
    """
    if len(hyps) != len(refs):
        raise ValueError(f"hyps and refs length mismatch: {len(hyps)} vs {len(refs)}")

    hyps_clean = _to_text_list(hyps, is_clean)
    refs_clean = _to_text_list(refs, is_clean)

    if scorer is None:
        scorer = F1RadGraph(reward_level=reward_level, model_type=model_type, cuda=cuda)
    mean_reward, _, _, _ = scorer(hyps=hyps_clean, refs=refs_clean)

    if reward_level == "all":
        simple, partial, complete = mean_reward
        return {
            "radgraph_simple": float(simple),
            "radgraph_partial": float(partial),
            "radgraph_complete": float(complete),
            "radgraph_f1": float(complete),
        }

    return {"radgraph_f1": float(mean_reward)}


def calc_radGraghF1_from_columns(
    df: pd.DataFrame,
    pred_col: str,
    gt_col: str = "reference_report_new",
    is_clean: bool = True,
    cuda: int = 0,
    model_type: str = "radgraph-xl",
) -> Mapping[str, float]:
    if pred_col not in df.columns:
        raise KeyError(f"Missing pred_col: {pred_col}")
    if gt_col not in df.columns:
        raise KeyError(f"Missing gt_col: {gt_col}")
    return calc_radGraghF1(
        df[pred_col].tolist(),
        df[gt_col].tolist(),
        is_clean=is_clean,
        cuda=cuda,
        model_type=model_type,
    )
