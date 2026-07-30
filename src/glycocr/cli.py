"""Typer CLI interface for GlycOCR."""

from pathlib import Path
import json
import logging

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn

console = Console()

app = typer.Typer(
    name="glycocr",
    help="GlycOCR CLI converting SNFG diagrams to IUPAC-condensed strings",
    rich_markup_mode="markdown",
)

def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True)]
    )


@app.command()
def predict(
    image: Path = typer.Option(
        ..., "--image", "-i", help="Path to SNFG image or directory"
    ),
    output_json: Path | None = typer.Option(
        None, "--output-json", "-o", help="Optional output JSON file path"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging")
) -> None:
    """🧠 Predict IUPAC string for an input SNFG diagram image."""
    setup_logging(verbose)
    from glycocr.inference.predictor import GlycOCR
    
    with console.status(f"[bold green]Predicting for image: {image}..."):
        predictor = GlycOCR.load_pretrained()
        try:
            result = predictor.predict(image)
        except Exception as e:
            console.print(f"[bold red]❌ Error predicting:[/bold red] {e}")
            raise typer.Exit(1)
            
    console.print(f"Result: [green]{result.iupac}[/green] (Valid: {result.is_valid})")
    if output_json:
        console.print(f"Output will be saved to: [cyan]{output_json}[/cyan]")
        with open(output_json, "w") as f:
            f.write(result.model_dump_json(exclude={'graph'}, indent=2))


@app.command()
def train(
    data_dir: Path = typer.Option(
        ..., "--data-dir", "-d", help="Path to directory containing binary dataset (images.bin, strings.bin, index.npz)"
    ),
    output_dir: Path = typer.Option(
        ..., "--output-dir", help="Output directory for trained model"
    ),
    epochs: int = typer.Option(3, "--epochs", "-e", help="Number of epochs to train"),
    batch_size: int = typer.Option(4, "--batch-size", "-b", help="Batch size"),
    lr: float = typer.Option(5e-4, "--lr", help="Learning rate"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging")
) -> None:
    """📈 Train or fine-tune GlycOCR model on binary SoA dataset."""
    setup_logging(verbose)
    console.print(f"Training model with dataset directory: [cyan]{data_dir}[/cyan]")
    console.print(f"Model output directory: [cyan]{output_dir}[/cyan]")
    
    from glycocr.models.model import GlycOCRModel
    from glycocr.data.dataset import GlycOCRDataset
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
    )
    
    console.print("[bold cyan]Starting training...[/bold cyan]")
    trainer.train()
    
    with console.status("[bold green]Saving model..."):
        model.save_pretrained(output_dir)
    console.print(f"✅ Training complete. Model saved to [cyan]{output_dir}[/cyan]")


@app.command()
def synthesize(
    iupac_list: Path = typer.Option(
        ..., "--iupac-list", help="Path to text file containing IUPAC strings"
    ),
    out_dir: Path = typer.Option(
        ..., "--out-dir", help="Output directory for generated dataset"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging")
) -> None:
    """🧪 Synthesize clean SNFG images and save them as a binary SoA dataset."""
    setup_logging(verbose)
    console.print(f"Synthesizing dataset from list: [cyan]{iupac_list}[/cyan]")
    console.print(f"Output directory: [cyan]{out_dir}[/cyan]")
    from glycocr.data.synthesizer import IUPACSynthesizer
    import numpy as np
    import io
    
    with open(iupac_list, 'r') as f:
        iupacs = [line.strip() for line in f if line.strip()]
    
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
        
        with open(images_bin_path, 'wb') as img_f, open(strings_bin_path, 'wb') as str_f:
            for i, iupac in enumerate(iupacs):
                try:
                    img = synth.synthesize(iupac)
                    
                    # Save string
                    str_bytes = iupac.encode('utf-8')
                    str_offset = str_f.tell()
                    str_f.write(str_bytes)
                    
                    str_offsets.append(str_offset)
                    str_lengths.append(len(str_bytes))
                    
                    # Save image as PNG bytes
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='PNG')
                    img_bytes = img_byte_arr.getvalue()
                    
                    img_offset = img_f.tell()
                    img_f.write(img_bytes)
                    
                    img_offsets.append(img_offset)
                    img_lengths.append(len(img_bytes))
                    
                except Exception as e:
                    console.print(f"[yellow]Failed to synthesize {iupac}:[/yellow] {e}")
                
                progress.advance(task_id, 1)
                
    # Save the Struct-of-Arrays (SoA) index
    np.savez(
        index_path,
        img_offsets=np.array(img_offsets, dtype=np.uint64),
        img_lengths=np.array(img_lengths, dtype=np.uint32),
        str_offsets=np.array(str_offsets, dtype=np.uint64),
        str_lengths=np.array(str_lengths, dtype=np.uint32)
    )
            
    console.print("✅ Synthesis complete. Binary dataset created.")


@app.command()
def fetch_dataset(
    output: Path = typer.Option(
        Path("dataset_iupac.txt"), "--output", "-o", help="Output text file path"
    ),
    synthetic_ratio: float = typer.Option(
        0.5, "--synthetic-ratio", help="Ratio of synthetic to real glycans to generate"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging")
) -> None:
    """📥 Fetch real glycans and generate synthetic variants for training."""
    setup_logging(verbose)
    import random
    import re
    try:
        from glycowork.glycan_data.loader import df_glycan
        from glycowork.motif.processing import canonicalize_iupac
    except ImportError:
        console.print("[red]❌ Please install glycowork:[/red] uv pip install glycowork")
        raise typer.Exit(code=1)

    console.print(f"Loading real glycans from glycowork database...")
    real_glycans = list(set(df_glycan['glycan'].dropna().tolist()))
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
        linkages = re.findall(r'[ab]\d-\d', mutated)
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
    
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        for g in hybrid_dataset:
            f.write(f"{g}\n")
            
    console.print(f"✅ [green]Successfully saved {len(hybrid_dataset)} total IUPAC strings to {output}[/green]")


def main() -> None:
    """Entrypoint function for CLI execution."""
    app()


if __name__ == "__main__":
    main()
