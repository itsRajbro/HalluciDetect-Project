"""Transformer baseline (Person 1): a conventional supervised reference.

A pretrained encoder (default: distilroberta-base) is fine-tuned end to end to
classify (question, candidate answer [, knowledge]) as factual / hallucinated.
It is NOT a Llama-based method: it sees only the text, and it is free to learn
surface cues such as answer length. Its job is to be the conventional reference
that SE and SEP are compared against, so its length-controlled results matter
more than its full-test numbers.

Protocol (enforced by code, not just convention)
------------------------------------------------
1. `select_and_freeze` uses TRAIN to fit and VALIDATION to choose
   (learning rate, epoch) and the decision threshold. It never loads the test
   split. It writes `frozen_config.json` (hyperparameters, threshold, checkpoint
   SHA-256, project config hash) and the validation detector run.
2. `predict_test_and_evaluate` refuses to run unless the frozen config exists,
   the checkpoint hash matches it, and the project `config_hash` is unchanged.
   It scores test ONCE with the frozen model and threshold, writes the test run,
   evaluates with the common evaluator, and drops `test_evaluated.json`.
3. After that marker exists, `select_and_freeze` refuses to run again unless
   `force=True`: re-selecting after seeing test results would break the protocol.

Design notes
------------
* Score = logit margin (logit[hallucinated] - logit[factual]), not a
  probability: probabilities saturate to 1.0 and create ties.
* Training uses bf16 autocast on CUDA; scoring runs in fp32 so margins are not
  quantised (bf16 logits would create many ties).
* The protocol logic is independent of PyTorch (a `Backend` object does the
  training/scoring), so it is unit-tested with a fake backend. `TorchBackend`
  is the real implementation.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Optional, Protocol, Sequence

import numpy as np

from src.common import evaluator as E
from src.common.config import REPO_ROOT, Config, get_config
from src.common.data import Sample, load_length_controlled, load_split
from src.common.schema import DetectorRecord, write_detector_output

DETECTOR = "transformer"
FROZEN_FILE = "frozen_config.json"
TEST_MARKER = "test_evaluated.json"
BEST_CKPT = "best.pt"
LOG_FILE = "training_log.json"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TransformerBaselineConfig:
    model_name: str = "distilroberta-base"
    include_knowledge: bool = True          # same input information as the Llama detectors (cfg.prompt.include_knowledge)
    max_length: int = 320
    batch_size: int = 16
    eval_batch_size: int = 64
    lr_grid: tuple = (2e-5, 5e-5)
    max_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 42
    threshold_criterion: str = "youden"
    train_subset: Optional[int] = None      # SMOKE RUNS ONLY: use the first N training samples

    def __post_init__(self):
        if self.train_subset is not None and (self.train_subset <= 0 or self.train_subset % 2):
            raise ValueError("train_subset must be a positive even number (keeps question pairs together)")
        if not self.lr_grid or self.max_epochs < 1:
            raise ValueError("need at least one learning rate and one epoch")

    def variant(self) -> str:
        short = re.sub(r"[^A-Za-z0-9_.\-]", "-", self.model_name.split("/")[-1])
        v = f"{short}_{'qak' if self.include_knowledge else 'qa'}"
        return v + "_smoke" if self.train_subset else v

    def to_dict(self) -> dict:
        d = asdict(self)
        d["lr_grid"] = list(self.lr_grid)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TransformerBaselineConfig":
        d = dict(d)
        d["lr_grid"] = tuple(d["lr_grid"])
        return cls(**d)


def default_out_dir(project_cfg: Config, cfg: TransformerBaselineConfig) -> Path:
    return REPO_ROOT / project_cfg.paths.outputs_dir / "transformer" / cfg.variant()


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def build_inputs(sample: Sample, include_knowledge: bool) -> tuple[str, Optional[str]]:
    """(segment A, segment B). A = question + candidate answer; B = knowledge (truncated first)."""
    a = f"Question: {sample.question.strip()}\nAnswer: {sample.answer.strip()}"
    b = f"Knowledge: {sample.knowledge.strip()}" if include_knowledge else None
    return a, b


# ---------------------------------------------------------------------------
# Backend interface (real: TorchBackend; tests: fake)
# ---------------------------------------------------------------------------
@dataclass
class EpochResult:
    epoch: int                       # 1-based
    val_scores: np.ndarray           # in the order of the validation samples passed to train()
    checkpoint: Path                 # weights after this epoch
    train_loss: Optional[float] = None
    seconds: Optional[float] = None          # whole epoch: training + validation scoring + checkpoint save
    val_seconds: Optional[float] = None      # validation scoring only


@dataclass
class ScoreResult:
    scores: np.ndarray
    latency_s_per_sample: float
    peak_gpu_mem_mb: Optional[float]


class Backend(Protocol):
    def train(self, cfg: TransformerBaselineConfig, lr: float, train_samples: Sequence[Sample],
              val_samples: Sequence[Sample], workdir: Path) -> list[EpochResult]: ...

    def score(self, cfg: TransformerBaselineConfig, checkpoint: Path,
              samples: Sequence[Sample]) -> ScoreResult: ...


# ---------------------------------------------------------------------------
# Protocol helpers (pure; unit-tested)
# ---------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pick_best(history: Sequence[dict]) -> int:
    """Index of the row with the highest validation AUROC (None/NaN count as worst; ties -> first)."""
    best_i, best_v = None, -np.inf
    for i, row in enumerate(history):
        v = row.get("val_auroc")
        v = -np.inf if v is None or not np.isfinite(v) else float(v)
        if best_i is None or v > best_v:
            best_i, best_v = i, v
    if best_i is None:
        raise ValueError("empty training history")
    return best_i


def assert_selection_allowed(out_dir: Path, force: bool) -> None:
    marker = Path(out_dir) / TEST_MARKER
    if marker.exists() and not force:
        raise RuntimeError(
            f"{marker} exists: the test set has already been evaluated with this baseline. "
            f"Re-running selection now would tune on test results and break the protocol. "
            f"Use force=True / --force only if you accept that and will report it.")


def check_frozen(out_dir: Path, project_config_hash: str) -> dict:
    """Load frozen_config.json and verify it is intact and still valid. Raises on any problem."""
    out_dir = Path(out_dir)
    fp = out_dir / FROZEN_FILE
    if not fp.is_file():
        raise FileNotFoundError(f"{fp} not found: run select_and_freeze (validation-only) before touching test")
    frozen = json.loads(fp.read_text(encoding="utf-8"))
    ckpt = out_dir / frozen["checkpoint"]["file"]
    if not ckpt.is_file():
        raise FileNotFoundError(f"frozen checkpoint {ckpt} is missing")
    if _sha256(ckpt) != frozen["checkpoint"]["sha256"]:
        raise ValueError("checkpoint does not match the frozen config (file changed after freezing)")
    if frozen["project"]["config_hash"] != project_config_hash:
        raise ValueError(
            f"project config_hash changed since freezing ({frozen['project']['config_hash']} -> "
            f"{project_config_hash}); re-run select_and_freeze")
    if frozen["threshold"]["selected_on"] != "validation":
        raise ValueError("frozen threshold was not selected on validation")
    return frozen


def _records(samples, split, scores, variant, project_cfg, latency_s, peak_gpu_mem_mb):
    return [DetectorRecord(sample_id=s.sample_id, split=split, detector=DETECTOR, variant=variant,
                           score=float(sc), config_hash=project_cfg.config_hash(),
                           prompt_version=project_cfg.prompt.version, latency_s=latency_s,
                           n_generations=0, peak_gpu_mem_mb=peak_gpu_mem_mb)
            for s, sc in zip(samples, scores)]


def _lib_versions() -> dict:
    out = {}
    for name in ("torch", "transformers", "tokenizers"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


# ---------------------------------------------------------------------------
# Phase 1: select on validation and freeze
# ---------------------------------------------------------------------------
def select_and_freeze(
    cfg: TransformerBaselineConfig,
    out_dir: Optional[Path] = None,
    *,
    project_cfg: Optional[Config] = None,
    backend: Optional[Backend] = None,
    train_samples: Optional[Sequence[Sample]] = None,
    val_samples: Optional[Sequence[Sample]] = None,
    runs_dir: Optional[Path] = None,
    force: bool = False,
) -> dict:
    """Train on train, choose (lr, epoch) and the threshold on validation, freeze. Never touches test."""
    project_cfg = project_cfg or get_config()
    out_dir = Path(out_dir) if out_dir else default_out_dir(project_cfg, cfg)
    assert_selection_allowed(out_dir, force)
    backend = backend or TorchBackend()

    train = list(train_samples) if train_samples is not None else load_split("train")
    val = list(val_samples) if val_samples is not None else load_split("validation")
    if cfg.train_subset:
        train = train[: cfg.train_subset]
    y_val = np.array([s.label for s in val], dtype=int)

    out_dir.mkdir(parents=True, exist_ok=True)
    workdir = out_dir / "epochs"
    shutil.rmtree(workdir, ignore_errors=True)
    history: list[dict] = []
    store: dict[tuple, EpochResult] = {}
    for lr in cfg.lr_grid:
        results = backend.train(cfg, lr, train, val, workdir / f"lr{lr:g}")
        for r in results:
            scores = np.asarray(r.val_scores, dtype=float)
            if scores.shape != (len(val),) or not np.isfinite(scores).all():
                raise ValueError(f"backend returned invalid validation scores for lr={lr}, epoch={r.epoch}")
            history.append({"lr": lr, "epoch": r.epoch, "val_auroc": E.auroc(y_val, scores),
                            "train_loss": r.train_loss, "seconds": r.seconds,
                            "val_seconds": r.val_seconds})
            store[(lr, r.epoch)] = r

    best = history[pick_best(history)]
    r = store[(best["lr"], best["epoch"])]
    reselected_after_test = (out_dir / TEST_MARKER).exists()      # only possible with force=True
    (out_dir / BEST_CKPT).unlink(missing_ok=True)
    shutil.move(str(r.checkpoint), str(out_dir / BEST_CKPT))
    shutil.rmtree(workdir, ignore_errors=True)               # drop every other epoch's weights

    val_scores = np.asarray(r.val_scores, dtype=float)
    thr = E.select_threshold(y_val, val_scores, cfg.threshold_criterion)

    variant = cfg.variant()
    path = None if runs_dir is None else Path(runs_dir) / "validation.jsonl"
    frozen = {
        "schema": 1,
        "detector": DETECTOR,
        "variant": variant,
        "baseline_config": cfg.to_dict(),
        "selected": {"lr": best["lr"], "epoch": best["epoch"], "val_auroc": best["val_auroc"]},
        "threshold": {"value": thr["value"], "criterion": thr["criterion"], "selected_on": "validation"},
        "checkpoint": {"file": BEST_CKPT, "sha256": _sha256(out_dir / BEST_CKPT)},
        "project": {"config_hash": project_cfg.config_hash(), "prompt_version": project_cfg.prompt.version},
        "reselected_after_test": reselected_after_test,
        "libraries": _lib_versions(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    val_path = write_detector_output(
        _records(val, "validation", val_scores, variant, project_cfg, None, None),
        path=path, threshold=thr["value"], cfg=project_cfg,
        description=f"transformer baseline {variant}: validation scores of the frozen model",
        extra={"frozen": {k: frozen[k] for k in ("baseline_config", "selected", "threshold")},
               "uses_llama_prompt": False})
    frozen["val_run_path"] = str(val_path)
    (out_dir / FROZEN_FILE).write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    (out_dir / LOG_FILE).write_text(json.dumps({"history": history}, indent=2), encoding="utf-8")
    return frozen


# ---------------------------------------------------------------------------
# Phase 2: final evaluation with the frozen configuration
# ---------------------------------------------------------------------------
def predict_test_and_evaluate(
    out_dir: Path,
    *,
    project_cfg: Optional[Config] = None,
    backend: Optional[Backend] = None,
    test_samples: Optional[Sequence[Sample]] = None,
    val_samples: Optional[Sequence[Sample]] = None,
    ctrl_samples: Optional[Sequence[Sample]] = None,
    include_length_controlled: bool = True,
    n_bins: int = 8,
    save: bool = True,
) -> dict:
    """Score test once with the frozen model + threshold and evaluate. Requires phase 1."""
    project_cfg = project_cfg or get_config()
    out_dir = Path(out_dir)
    frozen = check_frozen(out_dir, project_cfg.config_hash())
    cfg = TransformerBaselineConfig.from_dict(frozen["baseline_config"])
    backend = backend or TorchBackend()

    test = list(test_samples) if test_samples is not None else load_split("test")
    val = list(val_samples) if val_samples is not None else load_split("validation")
    sr = backend.score(cfg, out_dir / frozen["checkpoint"]["file"], test)
    scores = np.asarray(sr.scores, dtype=float)
    if scores.shape != (len(test),) or not np.isfinite(scores).all():
        raise ValueError("backend returned invalid test scores")

    # the test set has now been used: lock re-selection
    (out_dir / TEST_MARKER).write_text(json.dumps({
        "test_evaluated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint_sha256": frozen["checkpoint"]["sha256"]}, indent=2), encoding="utf-8")

    val_path = Path(frozen["val_run_path"])
    test_path = write_detector_output(
        _records(test, "test", scores, frozen["variant"], project_cfg, sr.latency_s_per_sample, sr.peak_gpu_mem_mb),
        path=val_path.parent / "test.jsonl", threshold=frozen["threshold"]["value"], cfg=project_cfg,
        description=f"transformer baseline {frozen['variant']}: test scores of the frozen model",
        extra={"frozen": {k: frozen[k] for k in ("baseline_config", "selected", "threshold")},
               "uses_llama_prompt": False})

    ctrl = ctrl_samples
    if ctrl is None and include_length_controlled:
        try:
            ctrl = load_length_controlled()
        except FileNotFoundError:
            ctrl = None
    return E.evaluate_run(val_path=val_path, test_path=test_path, val_samples=val, test_samples=test,
                          ctrl_samples=ctrl, include_length_controlled=False, n_bins=n_bins, save=save)


# ---------------------------------------------------------------------------
# Real backend (PyTorch + Hugging Face). Imports are lazy so the module (and the
# protocol tests) work without torch.
# ---------------------------------------------------------------------------
class TorchBackend:
    def __init__(self, device: Optional[str] = None, log_every: int = 50):
        self.device = device
        self.log_every = log_every

    # -- helpers -----------------------------------------------------------
    def _torch(self):
        import torch
        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        return torch, dev

    @staticmethod
    def _seed_all(torch, seed: int) -> None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _encode(tok, samples, cfg: TransformerBaselineConfig) -> list[dict]:
        pairs = [build_inputs(s, cfg.include_knowledge) for s in samples]
        a = [p[0] for p in pairs]
        if cfg.include_knowledge:
            b = [p[1] for p in pairs]
            try:   # truncate the knowledge first; never cut the question/answer segment
                enc = tok(a, b, truncation="only_second", max_length=cfg.max_length)
            except Exception:
                enc = tok(a, b, truncation="longest_first", max_length=cfg.max_length)
        else:
            enc = tok(a, truncation=True, max_length=cfg.max_length)
        keys = [k for k in ("input_ids", "attention_mask", "token_type_ids") if k in enc]
        return [{k: enc[k][i] for k in keys} for i in range(len(samples))]

    @staticmethod
    def _to_device(batch, dev):
        return {k: v.to(dev) for k, v in batch.items()}

    def _predict(self, torch, model, tok, feats, cfg, dev) -> np.ndarray:
        """Logit margins in fp32 (no autocast: bf16 logits would be quantised into ties)."""
        model.eval()
        n = len(feats)
        scores = np.zeros(n, dtype=float)
        order = sorted(range(n), key=lambda i: len(feats[i]["input_ids"]))   # less padding
        with torch.no_grad():
            for i in range(0, n, cfg.eval_batch_size):
                idx = order[i:i + cfg.eval_batch_size]
                batch = self._to_device(tok.pad([feats[j] for j in idx], return_tensors="pt"), dev)
                logits = model(**batch).logits.float()
                scores[idx] = (logits[:, 1] - logits[:, 0]).cpu().numpy()
        return scores

    # -- Backend interface ---------------------------------------------------
    def train(self, cfg, lr, train_samples, val_samples, workdir):
        import gc
        import math
        torch, dev = self._torch()
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

        self._seed_all(torch, cfg.seed)
        tok = AutoTokenizer.from_pretrained(cfg.model_name)
        tr = self._encode(tok, train_samples, cfg)
        va = self._encode(tok, val_samples, cfg)
        tr_labels = [s.label for s in train_samples]
        # fp32 master weights regardless of the checkpoint's stored dtype (bf16 is applied via autocast)
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name, num_labels=2, dtype=torch.float32).to(dev)

        no_decay = ("bias", "LayerNorm.weight")
        groups = [
            {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
             "weight_decay": cfg.weight_decay},
            {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
             "weight_decay": 0.0},
        ]
        opt = torch.optim.AdamW(groups, lr=lr)
        steps_per_epoch = math.ceil(len(tr) / cfg.batch_size)
        total = steps_per_epoch * cfg.max_epochs
        sched = get_linear_schedule_with_warmup(opt, int(cfg.warmup_ratio * total), total)
        use_bf16 = dev == "cuda" and torch.cuda.is_bf16_supported()

        if dev == "cuda":
            where = f"cuda ({torch.cuda.get_device_name(0)}), bf16 autocast={use_bf16}"
        else:
            where = "CPU -- NO GPU AVAILABLE, training will be very slow"
        print(f"[transformer] device: {where}; train samples={len(tr)}, steps/epoch={steps_per_epoch}, "
              f"batch={cfg.batch_size}, max_length={cfg.max_length}", flush=True)
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()

        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        results: list[EpochResult] = []
        for epoch in range(1, cfg.max_epochs + 1):
            model.train()
            t0 = time.perf_counter()
            g = torch.Generator().manual_seed(cfg.seed + epoch)
            perm = torch.randperm(len(tr), generator=g).tolist()
            losses = []
            for i in range(0, len(perm), cfg.batch_size):
                idx = perm[i:i + cfg.batch_size]
                batch = self._to_device(tok.pad([tr[j] for j in idx], return_tensors="pt"), dev)
                labels = torch.tensor([tr_labels[j] for j in idx], device=dev)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    logits = model(**batch).logits
                loss = torch.nn.functional.cross_entropy(logits.float(), labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                losses.append(loss.item())
                step = len(losses)
                if step % self.log_every == 0 or step == steps_per_epoch:
                    recent = float(np.mean(losses[-self.log_every:]))
                    print(f"[transformer] lr={lr:g} epoch {epoch}/{cfg.max_epochs} step {step}/{steps_per_epoch} "
                          f"loss {recent:.4f} elapsed {time.perf_counter() - t0:.0f}s", flush=True)
            t_val = time.perf_counter()
            val_scores = self._predict(torch, model, tok, va, cfg, dev)
            val_s = time.perf_counter() - t_val
            ckpt = workdir / f"epoch{epoch}.pt"
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, ckpt)
            mem = (f", peak GPU {torch.cuda.max_memory_allocated() / 2**20:.0f} MB" if dev == "cuda" else "")
            print(f"[transformer] epoch {epoch} done: train {t_val - t0:.0f}s, validation scoring {val_s:.0f}s{mem}",
                  flush=True)
            results.append(EpochResult(epoch=epoch, val_scores=val_scores, checkpoint=ckpt,
                                       train_loss=float(np.mean(losses)), seconds=time.perf_counter() - t0,
                                       val_seconds=val_s))
        del model, opt, sched
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return results

    def score(self, cfg, checkpoint, samples):
        torch, dev = self._torch()
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(cfg.model_name)
        feats = self._encode(tok, samples, cfg)
        model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=2, dtype=torch.float32)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        model.to(dev)
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        scores = self._predict(torch, model, tok, feats, cfg, dev)
        if dev == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 2**20 if dev == "cuda" else None
        return ScoreResult(scores=scores, latency_s_per_sample=elapsed / max(len(samples), 1), peak_gpu_mem_mb=peak)
