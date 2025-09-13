from __future__ import annotations

from . import utils
from . import report as reportmod
from . import report_train as trainlogmod
from . import config as cfgmod  # <-- needed for default_config()
import json
import typer
from .config import load_config, save_config, resolve_config
from .train import train as train_fn
from typing import Optional
from .info import model_info
from .eval import evaluate as eval_fn
from .report_train import generate_train_log_report
from . import infer as infermod   # <-- add this
from .report_train import generate_train_log_report

app = typer.Typer(add_completion=False, no_args_is_help=True, help="NeuNetwork CLI")

@app.command()
def init(config_path: str = "configs/default.yaml"):
    utils.ensure_dirs(["configs", "models", "runs", "data"])
    if utils.path_exists(config_path):
        typer.echo(f"Config already exists: {config_path}")
        raise typer.Exit(code=0)
    cfg = cfgmod.default_config()
    cfgmod.save_yaml(cfg, config_path)
    typer.echo(f"Created {config_path}")
    typer.echo("Folders ready: configs/, models/, runs/, data/")

@app.command()
def train(
        layers: str = typer.Option(None, help="Comma-separated hidden sizes, e.g. 128,64"),
        activations: str = typer.Option(None, help="Comma-separated activations, e.g. relu,relu"),
        epochs: int = typer.Option(None),
        batch_size: int = typer.Option(None),
        lr: float = typer.Option(None, help="Learning rate (Adam)"),
        dropout: float = typer.Option(None, min=0.0, max=0.99, help="Dropout probability"),
        weight_decay: float = typer.Option(None, min=0.0, help="L2 weight decay (Adam)"),
        seed: int = typer.Option(None),
        device: str = typer.Option(None, help="auto|cpu|cuda|mps"),
        config: str = typer.Option("configs/default.yaml"),
):
    cfg = load_config(config)

    if layers:       cfg["model"]["layers"] = [int(x) for x in layers.split(",") if x.strip()]
    if activations:  cfg["model"]["activations"] = [a.strip().lower() for a in activations.split(",")]
    if epochs is not None:       cfg["train"]["epochs"] = int(epochs)
    if batch_size is not None:   cfg["train"]["batch_size"] = int(batch_size)
    if lr is not None:           cfg["train"]["lr"] = float(lr)
    if dropout is not None:      cfg["model"]["dropout"] = float(dropout)           # ← NEW
    if weight_decay is not None: cfg["train"]["weight_decay"] = float(weight_decay) # ← NEW
    if seed is not None:         cfg["seed"] = int(seed)
    if device:                   cfg["device"] = device

    # Hard-lock to Adam; ignore legacy optimizer key if present
    cfg.get("train", {}).pop("optimizer", None)

    cfg = resolve_config(cfg)
    save_config(cfg, config)
    train_fn(cfg)

@app.command()
def eval(
        checkpoint: str = typer.Option("models/best.pt", help="Path to checkpoint"),
        metrics_out: str = typer.Option("runs/metrics.json", help="Where to write metrics JSON"),
        plots_out: str = typer.Option("runs/", help="Where to write plots"),
        config: str = typer.Option("configs/default.yaml", help="Config YAML"),
):
    cfg = load_config(config)
    cfg = resolve_config(cfg)
    res = eval_fn(cfg, checkpoint=checkpoint, metrics_out=metrics_out, plots_out=plots_out)
    typer.echo(json.dumps(res, indent=2, default=str))

@app.command()
def infer(
        images: Optional[str] = typer.Argument(None, help="Path to file or directory"),
        images_opt: Optional[str] = typer.Option(None, "--images", "-i", help="Alias for the same path"),
        checkpoint: str = typer.Option("models/best.pt"),
        out: str = typer.Option("runs/infer.json"),
        topk: int = typer.Option(3, min=1, max=10),
        recursive: bool = typer.Option(False, "--recursive/--no-recursive"),
):
    """Run inference on an image file or directory (top-K predictions)."""
    images_path = images_opt or images
    if not images_path:
        raise typer.BadParameter("Provide a path via POSITIONAL `IMAGES` or --images/-i.")
    infermod.infer(images_path, checkpoint, out, topk=topk, recursive=recursive)


@app.command("train-log")
def train_log_cmd(
        metrics: str = typer.Option("runs/metrics.json", help="Path to training metrics JSON"),
        out_md: str = typer.Option("runs/train_log.md", help="Where to write the training log"),
        eval_json: str | None = typer.Option(None, help="Optional JSON with test metrics"),
):
    res = generate_train_log_report(metrics_path=metrics, eval_path=eval_json, out_path=out_md)
    typer.echo(f"Training log written to: {res['path']}")


@app.command()
def report(
    infer_json: str = typer.Option("runs/infer.json", "--infer-json", "-j"),
    out: str = typer.Option("runs/infer_report.md", "--out", "-o"),
    label_regex: str = typer.Option(None, help="Optional regex with (?P<label>\\d) group to extract true label from filename"),
):
    res = reportmod.build_report(infer_json, out, label_regex=label_regex)
    typer.echo(f"Wrote {out}. Error rate: {res['summary']['error_rate']:.2%}")


@app.command("log-report")
def log_report_cmd(
        metrics: str = typer.Option("runs/metrics.json", "--metrics", "-m"),
        eval: str = typer.Option("runs/metrics_eval.json", "--eval", "-e"),
        out: str = typer.Option("runs/train_log.md", "--out", "-o"),
):
    """
    Generate final training report covering train/val/test metrics and verdict.
    """
    res = trainlogmod.generate_train_log_report(metrics_path=metrics, eval_path=eval, out_path=out)
    typer.echo(f"Wrote {out}. Verdict: {res['verdict']}")

@app.command()
def info(
        checkpoint: str = typer.Option("models/best.pt", help="Path to .pt ('' to skip)"),
        config: str = typer.Option("configs/default.yaml", help="YAML config"),
        out_md: str | None = typer.Option(None, help="If set, write a Markdown summary here"),
):
    res = model_info(config_path=config, checkpoint=(checkpoint or None), out_md=out_md)
    # pretty print to console
    typer.echo(json.dumps(res, indent=2, default=str))

if __name__ == "__main__":
    app()
