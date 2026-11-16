"""RadGraph-F1 Partial / Complete."""

from __future__ import annotations

import os
from typing import Dict, Sequence

# Prefer extracted RadGraph-XL (local CXR-BERT) over missing radgraph.tar.gz
RADGRAPH_XL_DIR = os.environ.get(
    "RADGRAPH_ARCHIVE",
    os.path.join(os.environ.get("DATA_ROOT", "./data"), "weights", "radgraph-xl"),
)


def _patch_transformers_trust():
    """AllenNLP embedder calls AutoModel/AutoTokenizer without trust_remote_code."""
    import transformers
    import transformers.dynamic_module_utils as dmu
    import radgraph.allennlp.common.cached_transformers as ct

    if getattr(ct, "_rg_eval_patched", False):
        return

    dmu.resolve_trust_remote_code = lambda *a, **k: True

    def _wrap(fn):
        def _inner(*args, **kwargs):
            kwargs["trust_remote_code"] = True
            return fn(*args, **kwargs)

        return _inner

    transformers.AutoModel.from_pretrained = _wrap(transformers.AutoModel.from_pretrained)
    transformers.AutoTokenizer.from_pretrained = _wrap(
        transformers.AutoTokenizer.from_pretrained
    )
    transformers.AutoConfig.from_pretrained = _wrap(transformers.AutoConfig.from_pretrained)
    ct.AutoModel = transformers.AutoModel

    # New transformers refuses encode_plus(add_special_tokens=False, return_token_type_ids=True)
    _encode_plus = transformers.PreTrainedTokenizerBase.encode_plus

    def _encode_plus_safe(self, *args, **kwargs):
        force_tti = False
        if kwargs.get("add_special_tokens") is False and kwargs.get("return_token_type_ids"):
            kwargs["return_token_type_ids"] = False
            force_tti = True
        out = _encode_plus(self, *args, **kwargs)
        if force_tti and "token_type_ids" not in out:
            n = len(out["input_ids"])
            try:
                out.data["token_type_ids"] = [0] * n
            except Exception:
                out["token_type_ids"] = [0] * n
        return out

    transformers.PreTrainedTokenizerBase.encode_plus = _encode_plus_safe
    ct._rg_eval_patched = True


def _make_scorer(cuda):
    _patch_transformers_trust()
    from radgraph import F1RadGraph
    from radgraph import utils as rg_utils
    from radgraph.radgraph import RadGraph

    # RadGraph-XL needs dataset name in each input doc
    _orig_preprocess = rg_utils.preprocess_reports

    def _preprocess_reports(report_list):
        out = _orig_preprocess(report_list)
        fixed = []
        for item in out:
            import json

            d = json.loads(item)
            d["dataset"] = "radgraph-xl"
            fixed.append(json.dumps(d))
        return fixed

    rg_utils.preprocess_reports = _preprocess_reports
    # RadGraph.forward imports preprocess_reports at call time from utils module attribute
    import radgraph.radgraph as rg_mod

    rg_mod.preprocess_reports = _preprocess_reports

    # DyGIE drops a report on RuntimeError, and postprocess then omits that doc_key.
    # F1RadGraph indexes every report id, so a skipped doc raises KeyError.
    _orig_post = rg_utils.postprocess_reports

    def _postprocess_reports(results):
        import json

        out = _orig_post(results)
        for raw in results:
            file = json.loads(raw)
            key = file.get("doc_key")
            if key not in out:
                out[key] = {
                    "text": "",
                    "entities": {},
                    "data_source": None,
                    "data_split": "inference",
                }
        return out

    rg_utils.postprocess_reports = _postprocess_reports

    class RadGraphLocal(RadGraph):
        def __init__(self, batch_size=1, cuda=None, **kwargs):
            import torch
            import torch.nn as nn
            from radgraph.allennlp.common.plugins import import_plugins
            from radgraph.allennlp.common.util import import_module_and_submodules
            from radgraph.allennlp.predictors.predictor import Predictor
            from radgraph.allennlp.models.archival import load_archive
            from radgraph.allennlp.common.checks import check_for_gpu

            nn.Module.__init__(self)
            if cuda is None:
                cuda = 0 if torch.cuda.is_available() else -1
            self.cuda = cuda
            self.batch_size = batch_size
            self.model_path = RADGRAPH_XL_DIR
            import_plugins()
            import_module_and_submodules("radgraph.dygie")
            check_for_gpu(self.cuda)
            archive = load_archive(
                self.model_path,
                weights_file=None,
                cuda_device=self.cuda,
                overrides="",
            )
            self.predictor = Predictor.from_archive(
                archive, predictor_name="dygie", dataset_reader_to_load="validation"
            )

        def forward(self, hyps):
            # Use patched preprocess
            import torch.nn as nn
            from radgraph.allennlp.commands.predict import _PredictManager
            from radgraph.utils import postprocess_reports

            assert isinstance(hyps, str) or isinstance(hyps, list)
            if isinstance(hyps, str):
                hyps = [hyps]
            hyps = ["None" if not s else s for s in hyps]
            model_input = _preprocess_reports(hyps)
            manager = _PredictManager(
                predictor=self.predictor,
                input_file=str(model_input),
                output_file=None,
                batch_size=self.batch_size,
                print_to_console=False,
                has_dataset_reader=True,
            )
            results = manager.run()
            return postprocess_reports(results)

    class F1RadGraphLocal(F1RadGraph):
        def __init__(self, reward_level, **kwargs):
            import torch.nn as nn

            nn.Module.__init__(self)
            assert reward_level in ["simple", "partial", "complete", "all"]
            self.reward_level = reward_level
            self.radgraph = RadGraphLocal(**kwargs)

    cuda_arg: int | bool = cuda if cuda is not None and int(cuda) >= 0 else -1
    return F1RadGraphLocal(reward_level="all", cuda=cuda_arg)


def compute_radgraph(
    refs: Sequence[str],
    hyps: Sequence[str],
    cuda: int = 0,
    model_type: str = "radgraph-xl",
) -> Dict[str, float]:
    refs_l = [str(r) if r is not None else "" for r in refs]
    hyps_l = [str(h) if h is not None else "" for h in hyps]
    scorer = _make_scorer(cuda)
    mean_reward, reward_list, _, _ = scorer(hyps=hyps_l, refs=refs_l)
    simple, partial, complete = mean_reward
    out = {
        "RadGraph-Partial": float(partial),
        "RadGraph-Complete": float(complete),
        "RadGraph-Simple": float(simple),
    }
    try:
        import numpy as np

        # reward_level="all" returns tuple of 3 lists -> array shape (3, n)
        # older path may return list of (s,p,c) -> (n, 3)
        if isinstance(reward_list, (tuple, list)) and len(reward_list) == 3 and not np.isscalar(reward_list[0]):
            simple_l, partial_l, complete_l = reward_list
            out["RadGraph_simple_per_sample"] = np.asarray(simple_l, dtype=float)
            out["RadGraph_partial_per_sample"] = np.asarray(partial_l, dtype=float)
            out["RadGraph_complete_per_sample"] = np.asarray(complete_l, dtype=float)
            out["RadGraph_combined_per_sample"] = out["RadGraph_complete_per_sample"]
        else:
            arr = np.asarray(reward_list, dtype=float)
            if arr.ndim == 2 and arr.shape[0] == 3 and arr.shape[1] != 3:
                out["RadGraph_simple_per_sample"] = arr[0]
                out["RadGraph_partial_per_sample"] = arr[1]
                out["RadGraph_complete_per_sample"] = arr[2]
                out["RadGraph_combined_per_sample"] = arr[2]
            elif arr.ndim == 2 and arr.shape[1] >= 3:
                out["RadGraph_combined_per_sample"] = arr[:, 2]
            elif arr.ndim == 1 and arr.shape[0] == len(refs_l):
                out["RadGraph_combined_per_sample"] = arr
    except Exception:
        pass
    return out
