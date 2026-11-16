"""Minimal NLG scoring helpers used by CE_calc (BLEU / METEOR / ROUGE / CIDEr)."""
from __future__ import annotations
import re
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
try:
    from pycocoevalcap.cider.cider import Cider
except Exception:  # optional
    Cider = None


def compute_NLG_scores(
    nlg_metrics: list[str],
    gen_sents_or_reports: list[str],
    ref_sents_or_reports: list[str],
) -> dict[str, float]:
    def convert_for_pycoco_scorer(sents_or_reports: list[str]):
        out = {}
        for num, text in enumerate(sents_or_reports):
            out[str(num)] = [re.sub(" +", " ", text.replace(".", " ."))]
        return out

    scorers = {}
    if "bleu" in nlg_metrics:
        scorers["bleu"] = Bleu(4)
    if "meteor" in nlg_metrics:
        scorers["meteor"] = Meteor()
    if "rouge" in nlg_metrics:
        scorers["rouge"] = Rouge()
    if "cider" in nlg_metrics:
        if Cider is None:
            raise ImportError("pycocoevalcap Cider is required for cider metric")
        scorers["cider"] = Cider()

    gens = convert_for_pycoco_scorer(gen_sents_or_reports)
    refs = convert_for_pycoco_scorer(ref_sents_or_reports)
    nlg_scores: dict[str, float] = {}
    for metric_name, scorer in scorers.items():
        score, _ = scorer.compute_score(refs, gens)
        if metric_name == "bleu":
            nlg_scores["bleu_1"] = score[0]
            nlg_scores["bleu_2"] = score[1]
            nlg_scores["bleu_3"] = score[2]
            nlg_scores["bleu_4"] = score[3]
        else:
            nlg_scores[metric_name] = score
    return nlg_scores
