# tests/test_cli_flow.py
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
from typer.testing import CliRunner

from neunet.cli import app  # Typer app

runner = CliRunner()


def _make_dummy_images(root: Path):
    """Create 28x28 grayscale PNGs under label folders so report can read true labels."""
    root.mkdir(parents=True, exist_ok=True)
    for label in (0, 3, 7):
        d = root / str(label)
        d.mkdir(parents=True, exist_ok=True)
        img = (np.random.rand(28, 28) * 255).astype("uint8")
        Image.fromarray(img, mode="L").save(d / f"sample_{label}.png")


@pytest.mark.usefixtures("patch_mnist")  # from tests/conftest.py
def test_cli_end_to_end(tmp_path, monkeypatch):
    # run inside an isolated temp workspace
    monkeypatch.chdir(tmp_path)

    # 1) init (creates configs/default.yaml, models/, runs/)
    res = runner.invoke(app, ["init"])
    assert res.exit_code == 0, res.output
    cfg = Path("configs/default.yaml")
    assert cfg.exists()

    # (optional) tweak a couple of config values for speed/consistency
    # Not required if you pass flags below, but helps if train reads YAML only.
    # We'll primarily drive via flags.

    # 2) train (with a tiny model for speed)
    res = runner.invoke(
        app,
        [
            "train",
            "--layers", "64",
            "--activations", "relu",
            "--epochs", "2",
            "--batch-size", "16",
            "--lr", "1e-3",
            "--dropout", "0.1",
            "--weight-decay", "1e-4",
        ],
    )
    assert res.exit_code == 0, res.output
    assert Path("models/best.pt").exists()
    assert Path("runs/metrics.json").exists()

    # 3) eval (final exam on test set)
    res = runner.invoke(app, ["eval", "--checkpoint", "models/best.pt"])
    assert res.exit_code == 0, res.output
    metrics = json.loads(Path("runs/metrics.json").read_text(encoding="utf-8"))
    assert "test_acc" in metrics
    assert Path("runs/confusion_matrix.png").exists() or Path("runs/confusion_matrix.json").exists()
    assert Path("runs/evaluation.md").exists()

    # 4) training log (renamed from "training report")
    res = runner.invoke(app, ["train-log"])
    assert res.exit_code == 0, res.output
    assert Path("runs/train_log.md").exists()

    # 5) infer on custom images
    imgs = Path("my_imgs2")
    _make_dummy_images(imgs)
    res = runner.invoke(
        app,
        [
            "infer",
            "--images", str(imgs),
            "--checkpoint", "models/best.pt",
            "--out", "runs/infer.json",
            "--topk", "3",
            "--config", "configs/default.yaml",
        ],
    )
    assert res.exit_code == 0, res.output
    infer_items = json.loads(Path("runs/infer.json").read_text(encoding="utf-8"))
    assert isinstance(infer_items, list) and len(infer_items) >= 3
    assert {"file", "pred"} <= set(infer_items[0].keys())

    # 6) inference report (CLI if present; fallback to function)
    # Some branches expose a "report" command; if not, call function directly.
    res = runner.invoke(app, ["report", "--in", "runs/infer.json", "--out", "runs/inference_report.md"])
    if res.exit_code != 0:
        # Fallback to Python API
        from neunet.report import build_report
        build_report("runs/infer.json", "runs/inference_report.md")
    assert Path("runs/inference_report.md").exists()
