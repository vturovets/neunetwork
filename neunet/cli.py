
from typing import Optional
import typer
from . import config as cfgmod
from . import train as trainmod
from . import eval as evalmod
from . import infer as infermod
from . import utils
from . import report as reportmod
from . import report_train as trainlogmod


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
        layers: str = typer.Option("128,64", help="Comma-separated hidden sizes, e.g., 128,64"),
        activations: str = typer.Option("relu,relu", help="Comma-separated activations per hidden layer"),
        epochs: int = typer.Option(5, min=1),
        batch_size: int = typer.Option(64, min=1),
        lr: float = typer.Option(1e-3, min=0.0),
        seed: int = typer.Option(42),
        device: str = typer.Option("auto", help="auto|cpu|cuda|mps"),
        val_split: float = typer.Option(0.0, min=0.0, max=0.5),
        config_path: str = typer.Option("configs/default.yaml"),
):
    """Train the model on MNIST and save checkpoints/metrics."""
    cfg = cfgmod.load_and_resolve(config_path,
                                  overrides=dict(
                                      model={"layers": [int(x) for x in layers.split(",") if x] ,
                                             "activations": [a.strip().lower() for a in activations.split(",") if a]},
                                      train={"epochs": epochs, "batch_size": batch_size, "lr": lr},
                                      seed=seed, device=device, data={"val_split": val_split},
                                  )
                                  )
    typer.echo(f"Using device: {utils.pick_device(cfg['device'])}")
    trainmod.train(cfg)  # TODO: implement

@app.command()
def eval(
        checkpoint: str = typer.Option("models/best.pt"),
        metrics_out: str = typer.Option("runs/metrics.json"),
        plots_out: str = typer.Option("runs/"),
        config_path: str = typer.Option("configs/default.yaml"),
):
    """Evaluate test accuracy and test loss; write metrics and plots."""
    cfg = cfgmod.load_and_resolve(config_path)
    evalmod.evaluate(cfg, checkpoint, metrics_out, plots_out)  # TODO: implement

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

if __name__ == "__main__":
    app()
