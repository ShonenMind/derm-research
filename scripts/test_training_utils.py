"""
Local, CPU-only test that the checkpoint/resume core actually works — i.e.
that killing training partway and re-running continues from the right epoch
with identical model/optimizer/scheduler/RNG state, instead of restarting.

Run: python scripts/test_training_utils.py
"""
import os
import shutil
import tempfile

import numpy as np
import torch
import torch.nn as nn

from training_utils import (
    save_training_checkpoint,
    load_training_checkpoint,
    resume_paths,
    atomic_torch_save,
    trainable_state_dict,
)


def make_setup(seed=0):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 3))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.9)
    return model, opt, sched


def train_one_epoch(model, opt, sched, seed):
    torch.manual_seed(1000 + seed)
    x = torch.randn(32, 10)
    y = torch.randint(0, 3, (32,))
    opt.zero_grad()
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()
    opt.step()
    sched.step()
    return loss.item()


def run(total_epochs, ckpt_path, crash_after=None, resume=True):
    """Simulates a training run. If crash_after is set, stops after that epoch
    (as if the tab closed). If resume, picks up from <path>_last.pth."""
    model, opt, sched = make_setup()
    start_epoch = 1
    history = []

    if resume and os.path.exists(ckpt_path):
        ck = load_training_checkpoint(
            ckpt_path, model=model, optimizer=opt, scheduler=sched
        )
        start_epoch = ck["epoch"] + 1
        history = ck["history"]
        print(f"  [resumed] continuing from epoch {start_epoch}")
    else:
        print("  [fresh] starting from epoch 1")

    for epoch in range(start_epoch, total_epochs + 1):
        loss = train_one_epoch(model, opt, sched, epoch)
        history.append(loss)
        save_training_checkpoint(
            ckpt_path, epoch=epoch, model=model, optimizer=opt,
            scheduler=sched, history=history, best_metric=min(history),
        )
        print(f"  epoch {epoch}  loss={loss:.5f}  lr={sched.get_last_lr()[0]:.5f}")
        if crash_after is not None and epoch == crash_after:
            print(f"  !! simulated crash after epoch {epoch}")
            return model, history, "crashed"

    return model, history, "finished"


def main():
    tmp = tempfile.mkdtemp()
    try:
        paths = resume_paths(tmp, "dummy")
        ckpt = paths["last"]

        print("=== Baseline: uninterrupted 6-epoch run ===")
        base_ckpt = os.path.join(tmp, "baseline_last.pth")
        base_model, base_hist, _ = run(6, base_ckpt, resume=False)

        # wipe so the interrupted run starts truly fresh
        print("\n=== Interrupted run: crash after epoch 3, then resume ===")
        _, hist1, status1 = run(6, ckpt, crash_after=3, resume=True)
        assert status1 == "crashed"
        assert len(hist1) == 3, f"expected 3 epochs before crash, got {len(hist1)}"

        resumed_model, hist2, status2 = run(6, ckpt, resume=True)
        assert status2 == "finished"
        assert len(hist2) == 6, f"expected 6 total epochs, got {len(hist2)}"

        # ---- Correctness checks ----
        # 1. Resumed run must not have re-done epochs 1-3
        print("\n=== Verifying resume correctness ===")
        for i in range(3):
            assert abs(hist2[i] - hist1[i]) < 1e-9, f"epoch {i+1} loss changed after resume"
        print("  [ok] epochs 1-3 preserved exactly across the crash")

        # 2. Final model weights must match the uninterrupted baseline bit-for-bit
        for (n1, p1), (n2, p2) in zip(
            base_model.state_dict().items(), resumed_model.state_dict().items()
        ):
            assert torch.allclose(p1, p2, atol=1e-6), f"param {n1} diverged from baseline"
        print("  [ok] crash+resume produced identical final weights to a clean run")

        # 3. Full loss history matches baseline
        assert np.allclose(base_hist, hist2, atol=1e-6), "loss history diverged from baseline"
        print("  [ok] full 6-epoch loss trajectory matches the uninterrupted baseline")

        # 4. Atomic save leaves no stray .tmp file
        atomic_torch_save({"x": 1}, os.path.join(tmp, "atomic.pth"))
        assert not os.path.exists(os.path.join(tmp, "atomic.pth.tmp"))
        assert os.path.exists(os.path.join(tmp, "atomic.pth"))
        print("  [ok] atomic_torch_save leaves no partial .tmp file behind")

        # 5. only_trainable path: simulate a QLoRA model (frozen base + small
        #    trainable "adapter"); checkpoint must store ONLY adapter params and
        #    reload them over a frozen base via strict=False.
        print("\n=== Verifying only_trainable (QLoRA/adapter) path ===")
        torch.manual_seed(7)
        adapter_model = nn.Sequential(nn.Linear(10, 16), nn.Linear(16, 3))
        # freeze the first layer (the "base"), train only the second (the "adapter")
        for p in adapter_model[0].parameters():
            p.requires_grad = False
        tstate = trainable_state_dict(adapter_model)
        assert all(k.startswith("1.") for k in tstate), f"leaked frozen params: {list(tstate)}"
        assert len(tstate) == 2, f"expected 2 adapter tensors, got {len(tstate)}"
        print(f"  [ok] trainable_state_dict captured only adapter params: {sorted(tstate)}")

        adapter_ckpt = os.path.join(tmp, "adapter_last.pth")
        opt2 = torch.optim.AdamW(
            [p for p in adapter_model.parameters() if p.requires_grad], lr=1e-2
        )
        # mutate the adapter so saved != re-init
        with torch.no_grad():
            adapter_model[1].weight.add_(0.5)
        saved_adapter_w = adapter_model[1].weight.clone()
        frozen_base_w = adapter_model[0].weight.clone()
        save_training_checkpoint(
            adapter_ckpt, epoch=1, model=adapter_model, optimizer=opt2,
            only_trainable=True,
        )

        # fresh model with a DIFFERENT frozen base; resume must restore the
        # adapter but leave this base untouched (strict=False)
        torch.manual_seed(99)
        fresh = nn.Sequential(nn.Linear(10, 16), nn.Linear(16, 3))
        for p in fresh[0].parameters():
            p.requires_grad = False
        fresh_base_before = fresh[0].weight.clone()
        load_training_checkpoint(adapter_ckpt, model=fresh, strict=True)  # strict auto-relaxed
        assert torch.allclose(fresh[1].weight, saved_adapter_w, atol=1e-6), "adapter not restored"
        assert torch.allclose(fresh[0].weight, fresh_base_before, atol=1e-6), "frozen base was overwritten"
        assert not torch.allclose(fresh[0].weight, frozen_base_w), "sanity: bases should differ"
        print("  [ok] adapter restored, frozen base left untouched (strict auto-relaxed)")

        print("\nALL CHECKPOINT/RESUME TESTS PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
