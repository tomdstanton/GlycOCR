"""Typer CLI interface for GlycOCR."""

import importlib.metadata
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

console = Console()

try:
    _metadata = importlib.metadata.metadata("glycocr")
    _summary = _metadata.get("Summary", "glycocr CLI")  # type: ignore
except Exception:
    _summary = "glycocr CLI"

app = typer.Typer(
    name="glycocr",
    help=_summary,
    rich_markup_mode="markdown",
    context_settings={"help_option_names": ["-h", "--help"]},
)


def version_callback(value: bool) -> None:
    """Show the version and exit."""
    if value:
        try:
            version = importlib.metadata.version("glycocr")
            console.print(f"[bold]glycocr v{version}[/bold] - {_summary}")
        except Exception:
            console.print("glycocr (unknown version)")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None, "--version", "-v", callback=version_callback, is_eager=True, help="Show the version and exit."
    ),
    verbose: int = typer.Option(0, "--verbose", "-V", count=True, help="Increase verbosity level"),
) -> None:
    """Global configuration for the CLI."""
    setup_logging(verbose)


def setup_logging(verbose: int) -> None:
    """Configure the global logger based on verbosity level."""
    if verbose == 0:
        level = logging.WARNING
    elif verbose == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True)],
    )
    logging.getLogger("markdown_it").setLevel(logging.INFO)


from transformers import TrainerCallback


class RichProgressCallback(TrainerCallback):
    """Custom Hugging Face Trainer callback for rich progress bars."""
    def __init__(self, console, epochs):
        self.console = console
        self.progress = None
        self.task_id = None
        self.epochs = epochs

    def on_train_begin(self, args, state, control, **kwargs):
        from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("Loss: {task.fields[loss]:.4f}"),
            console=self.console,
            transient=False,
        )
        self.progress.start()
        self.task_id = self.progress.add_task(f"Training Epoch 1/{self.epochs}", total=state.max_steps, loss=0.0)

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch = int(state.epoch) if state.epoch else 0
        if self.progress and self.task_id is not None:
            self.progress.update(self.task_id, description=f"Training Epoch {epoch + 1}/{self.epochs}")
            
    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.progress and self.task_id is not None and logs and "loss" in logs:
            self.progress.update(self.task_id, loss=logs["loss"])

    def on_step_end(self, args, state, control, **kwargs):
        if self.progress and self.task_id is not None:
            self.progress.update(self.task_id, advance=1)

    def on_train_end(self, args, state, control, **kwargs):
        if self.progress:
            self.progress.stop()


@app.command()
def infer(
    image: typer.FileBinaryRead = typer.Argument("-", help="Path to SNFG image or `-` for stdin"),
    output_json: typer.FileTextWrite = typer.Argument("-", help="Output JSON file path or `-` for stdout"),
) -> None:
    """:brain: Predict IUPAC string for an input SNFG diagram image."""
    import warnings

    from transformers.utils import logging as hf_logging
    
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.modeling_attn_mask_utils")
    warnings.filterwarnings("ignore", message=".*use_return_dict is deprecated.*")
    warnings.filterwarnings("ignore", message=".*image_processor_class = 'CLIPImageProcessor'.*")
    hf_logging.set_verbosity_error()

    from glycocr.inference.predictor import GlycOCR

    with console.status("[bold green]Predicting for image..."):
        predictor = GlycOCR.load_pretrained()
        try:
            import torch
            import torchvision

            img_bytes = image.read()
            img_tensor = torch.frombuffer(bytearray(img_bytes), dtype=torch.uint8)
            img_tensor = torchvision.io.decode_image(img_tensor)
            result = predictor.predict(img_tensor)
        except Exception as e:
            console.print(f"[bold red]:x: Error predicting:[/bold red] {e}")
            raise typer.Exit(1)

    if output_json.name != "<stdout>":
        console.print(f"Result: [green]{result.iupac}[/green] (Valid: {result.is_valid})")
        console.print(f"Output saved to: [cyan]{output_json.name}[/cyan]")

    from dataclasses import asdict

    import orjson

    result_dict = asdict(result)
    result_dict.pop("graph", None)
    output_json.write(orjson.dumps(result_dict).decode("utf-8") + "\n")


@app.command()
def train(
    data_dir: Path = typer.Argument(
        ..., help="Path to directory containing binary dataset (images.bin, strings.bin, index.npz)"
    ),
    output_dir: Path = typer.Argument(..., help="Output directory for trained model"),
    epochs: int = typer.Option(3, "--epochs", "-e", help="Number of epochs to train"),
    batch_size: int = typer.Option(4, "--batch-size", "-b", help="Batch size"),
    lr: float = typer.Option(5e-4, "--lr", help="Learning rate"),
    resume: bool = typer.Option(False, "--resume", help="Resume from latest checkpoint in output directory"),
) -> None:
    """:chart_with_upwards_trend: Train or fine-tune GlycOCR model on binary SoA dataset."""
    import warnings
    from transformers.utils import logging as hf_logging

    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.modeling_attn_mask_utils")
    warnings.filterwarnings("ignore", message=".*use_return_dict is deprecated.*")
    warnings.filterwarnings("ignore", message=".*image_processor_class = 'CLIPImageProcessor'.*")
    hf_logging.set_verbosity_error()

    console.print(f"Training model with dataset directory: [cyan]{data_dir}[/cyan]")
    console.print(f"Model output directory: [cyan]{output_dir}[/cyan]")

    from glycocr.data.dataset import GlycOCRDataset
    from glycocr.models.model import GlycOCRModel
    from glycocr.training.trainer import GlycOCRTrainer

    with console.status("[bold green]Loading model and processor..."):
        model = GlycOCRModel()

    dataset = GlycOCRDataset(data_dir=data_dir, processor=model.processor)
    console.print(f"Loaded [green]{len(dataset)}[/green] samples from binary dataset.")

    trainer = GlycOCRTrainer(
        model=model,
        train_dataset=dataset,
        output_dir=str(output_dir),
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        disable_tqdm=True,  # Turn off standard HF progress bar
    )
    
    # Add custom rich callback
    trainer.extra_kwargs["callbacks"] = [RichProgressCallback(console, epochs)]

    console.print("[bold cyan]Starting training...[/bold cyan]")
    trainer.train(resume_from_checkpoint=resume)

    with console.status("[bold green]Saving model..."):
        model.save_pretrained(output_dir)
    console.print(f":white_check_mark: Training complete. Model saved to [cyan]{output_dir}[/cyan]")


prep_app = typer.Typer(help=":wrench: Data preparation tools")
app.add_typer(prep_app, name="prep")


@prep_app.command("synthesize")
def prep_synthesize(
    iupac_list: typer.FileText = typer.Argument(
        "-", help="Path to text file containing IUPAC strings or `-` for stdin"
    ),
    out_dir: Path = typer.Argument(..., help="Output directory for generated dataset"),
) -> None:
    """:test_tube: Synthesize clean SNFG images and save them as a binary SoA dataset."""
    console.print(f"Synthesizing dataset from list: [cyan]{iupac_list}[/cyan]")
    console.print(f"Output directory: [cyan]{out_dir}[/cyan]")

    import numpy as np

    from glycocr.data.synthesizer import IUPACSynthesizer

    iupacs = [line.strip() for line in iupac_list if line.strip()]
    synth = IUPACSynthesizer()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    images_bin_path = out_path / "images.bin"
    strings_bin_path = out_path / "strings.bin"
    index_path = out_path / "index.npz"

    img_offsets = []
    img_lengths = []
    str_offsets = []
    str_lengths = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        task_id = progress.add_task("Synthesizing dataset...", total=len(iupacs))

        with open(images_bin_path, "wb") as img_f, open(strings_bin_path, "wb") as str_f:
            for i, iupac in enumerate(iupacs):
                try:
                    img = synth.synthesize(iupac)

                    str_bytes = iupac.encode("utf-8")
                    str_offset = str_f.tell()
                    str_f.write(str_bytes)

                    str_offsets.append(str_offset)
                    str_lengths.append(len(str_bytes))

                    import torchvision

                    img_bytes = torchvision.io.encode_png(img).numpy().tobytes()

                    img_offset = img_f.tell()
                    img_f.write(img_bytes)

                    img_offsets.append(img_offset)
                    img_lengths.append(len(img_bytes))

                except Exception as e:
                    console.print(f"[yellow]Failed to synthesize {iupac}:[/yellow] {e}")

                progress.advance(task_id, 1)

    np.savez(
        index_path,
        img_offsets=np.array(img_offsets, dtype=np.uint64),
        img_lengths=np.array(img_lengths, dtype=np.uint32),
        str_offsets=np.array(str_offsets, dtype=np.uint64),
        str_lengths=np.array(str_lengths, dtype=np.uint32),
    )

    console.print(":white_check_mark: Synthesis complete. Binary dataset created.")


@prep_app.command("fetch")
def prep_fetch(
    output: typer.FileTextWrite = typer.Argument("-", help="Output text file path or `-` for stdout"),
    synthetic_ratio: float = typer.Option(
        0.5, "--synthetic-ratio", help="Ratio of synthetic to real glycans to generate"
    ),
) -> None:
    """:inbox_tray: Fetch real glycans and generate synthetic variants for training."""
    import random
    import re

    try:
        from glycowork.glycan_data.loader import df_glycan
        from glycowork.motif.processing import canonicalize_iupac
    except ImportError:
        console.print("[red]:x: Please install glycowork:[/red] uv pip install glycowork")
        raise typer.Exit(code=1)

    console.print("Loading real glycans from glycowork database...")
    real_glycans = list(set(df_glycan["glycan"].dropna().tolist()))
    console.print(f"Found [cyan]{len(real_glycans)}[/cyan] unique real IUPAC sequences.")

    valid_real_glycans = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        canon_task = progress.add_task("Canonicalizing sequences...", total=len(real_glycans))
        for g in real_glycans:
            try:
                valid_real_glycans.append(canonicalize_iupac(g))
            except Exception:
                pass
            progress.advance(canon_task, 1)

    valid_real_glycans = list(set(valid_real_glycans))
    console.print(f"Canonicalized [cyan]{len(valid_real_glycans)}[/cyan] real sequences.")

    num_synthetic = int(len(valid_real_glycans) * synthetic_ratio)
    console.print(f"Generating [cyan]{num_synthetic}[/cyan] synthetic/randomized variants...")

    residue_subs = {
        "Gal": ["Glc", "Man", "GalNAc"],
        "Glc": ["Gal", "Man", "GlcNAc"],
        "Man": ["Gal", "Glc", "Fuc"],
        "GlcNAc": ["GalNAc", "ManNAc"],
        "GalNAc": ["GlcNAc", "ManNAc"],
        "Neu5Ac": ["Neu5Gc", "KdN"],
        "Fuc": ["Rha", "Qui"],
    }
    linkage_subs = ["a1-2", "a1-3", "a1-4", "a1-6", "b1-2", "b1-3", "b1-4", "b1-6", "a2-3", "a2-6", "a2-8"]

    def randomize_glycan(iupac: str) -> str:
        mutated = iupac
        linkages = re.findall(r"[ab]\d-\d", mutated)
        for link in linkages:
            if random.random() < 0.1:
                mutated = mutated.replace(link, random.choice(linkage_subs), 1)
        for res, subs in residue_subs.items():
            if res in mutated and random.random() < 0.1:
                mutated = mutated.replace(res, random.choice(subs), 1)
        return mutated

    synthetic_glycans = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        synth_task = progress.add_task("Synthesizing variants...", total=num_synthetic)
        for _ in range(num_synthetic):
            base_glycan = random.choice(valid_real_glycans)
            synthetic_glycans.append(randomize_glycan(base_glycan))
            progress.advance(synth_task, 1)

    hybrid_dataset = list(set(valid_real_glycans + synthetic_glycans))
    random.shuffle(hybrid_dataset)

    if output.name != "<stdout>":
        Path(output.name).parent.mkdir(parents=True, exist_ok=True)

    for g in hybrid_dataset:
        output.write(f"{g}\n")

    if output.name != "<stdout>":
        console.print(
            f":white_check_mark: [green]Successfully saved {len(hybrid_dataset)} total IUPAC strings to {output.name}[/green]"
        )


@app.command()
def deploy(
    repo_id: str = typer.Argument(..., help="Target Hugging Face repository ID (e.g. username/glycocr)"),
    model_path: str = typer.Argument(..., help="Path to the model file or directory"),
    token: str = typer.Option(None, help="HF API token. Falls back to HF_TOKEN env var if not set."),
) -> None:
    """:rocket: Deploy a trained model to the Hugging Face Hub."""
    from glycocr.deploy import deploy_to_huggingface

    try:
        with console.status(f"[bold green]Deploying {model_path} to {repo_id}..."):
            deploy_to_huggingface(repo_id=repo_id, model_path=model_path, token=token)
        console.print(":white_check_mark: Deployment complete!")
    except Exception as e:
        console.print(f"[bold red]:x: Deployment failed:[/bold red] {e}")
        raise typer.Exit(1)
