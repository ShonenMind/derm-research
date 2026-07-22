"""
Checkpoint / resume utilities for the BCN20000 benchmark notebook.

The point of this module: make training **interruption-safe**. Colab (even
Pro+) can drop your session when a tab closes or a runtime recycles. Without
resumable checkpoints, a dropped session means the whole run — and the
compute credits it cost — is gone. These helpers let a training loop:

  1. Save a full "last" checkpoint every epoch (model + optimizer + scheduler
     + scaler + epoch counter + best-metric + early-stop counter + RNG state),
     written atomically so a mid-write disconnect can't corrupt it.
  2. On restart, detect that "last" checkpoint and resume from the next epoch
     with all state restored — so re-running the training cell continues
     instead of starting over.

Everything is plain PyTorch/numpy with no notebook globals, so it can be
unit-tested locally (see scripts/test_training_utils.py) before spending any
Colab GPU time on it.
"""
from __future__ import annotations

import os
import random
from typing import Any, Optional

import numpy as np
import torch


def atomic_torch_save(obj: Any, path: str) -> str:
    """
    torch.save that can't leave a half-written (corrupt) file if the process
    dies mid-write — which matters a lot when the target is Google Drive and
    the "process dying" is a Colab tab closing. Writes to a temp file on the
    same directory, then atomically renames it into place.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)  # atomic on the same filesystem
    return path


def _capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def _restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def trainable_state_dict(model) -> dict:
    """
    Only the parameters that actually get trained. For a QLoRA VLM that's just
    the LoRA adapter (a few MB) — saving the full state_dict would also dump
    the frozen 4-bit base model (many GB), which is wasteful and, for
    quantized weights, not cleanly reloadable anyway.
    """
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v for k, v in model.state_dict().items() if k in trainable}


def save_training_checkpoint(
    path: str,
    *,
    epoch: int,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    best_metric: Optional[float] = None,
    epochs_no_improve: int = 0,
    history: Optional[dict] = None,
    extra: Optional[dict] = None,
    save_rng: bool = True,
    only_trainable: bool = False,
) -> str:
    """
    Persist everything needed to resume training after `epoch` completed.
    `extra` lets callers stash model-specific fields (e.g. class names).
    Set only_trainable=True for QLoRA/adapter models to store just the
    trainable params (keeps the checkpoint small; pair with strict=False on
    load so the frozen base is left untouched).
    """
    model_state = trainable_state_dict(model) if only_trainable else model.state_dict()
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_metric": best_metric,
        "epochs_no_improve": epochs_no_improve,
        "history": history,
        "only_trainable": only_trainable,
        "rng_state": _capture_rng_state() if save_rng else None,
    }
    if extra:
        ckpt.update(extra)
    return atomic_torch_save(ckpt, path)


def load_training_checkpoint(
    path: str,
    *,
    model=None,
    optimizer=None,
    scheduler=None,
    scaler=None,
    map_location="cpu",
    restore_rng: bool = True,
    strict: bool = True,
) -> dict:
    """
    Load a checkpoint written by save_training_checkpoint and restore state
    into whichever of model/optimizer/scheduler/scaler are passed. Returns the
    raw checkpoint dict so the caller can read epoch/best_metric/history/etc.
    Pass strict=False for adapter/only_trainable checkpoints, so the stored
    LoRA weights load over a base model whose other keys aren't in the file.
    """
    ckpt = torch.load(path, map_location=map_location)
    # A checkpoint that only stored trainable params must load non-strictly.
    effective_strict = strict and not ckpt.get("only_trainable", False)
    if model is not None and ckpt.get("model_state_dict") is not None:
        model.load_state_dict(ckpt["model_state_dict"], strict=effective_strict)
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    if restore_rng and ckpt.get("rng_state") is not None:
        _restore_rng_state(ckpt["rng_state"])
    return ckpt


def resume_paths(checkpoint_dir: str, model_key: str) -> dict:
    """
    Standard checkpoint file layout for a model, all under its results dir:
      <dir>/<key>_best.pth   — best-val weights only (for evaluation)
      <dir>/<key>_last.pth   — full state after the most recent epoch (resume)
    """
    return {
        "best": os.path.join(checkpoint_dir, f"{model_key}_best.pth"),
        "last": os.path.join(checkpoint_dir, f"{model_key}_last.pth"),
    }
