import torch
import torchvision
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from glycocr.inference.predictor import GlycOCR
from glycocr.data.synthesizer import IUPACSynthesizer
import typer

app = typer.Typer(help="Visualize GlycOCR predictions by synthesizing the output back to SNFG.")

@app.command()
def main(
    image_path: str = typer.Argument(..., help="Path to the input SNFG PNG image"),
    model_path: str = typer.Option("tomdstanton/GlycOCR", "--model", "-m", help="Path to local trained weights or HF repo")
):
    print(f"Loading model from {model_path}...")
    predictor = GlycOCR.load_pretrained(model_path)
    
    print("Running OCR inference...")
    img_tensor = torchvision.io.decode_image(image_path)
    result = predictor.predict(img_tensor)
    
    predicted_iupac = result.iupac
    print(f"\nPredicted IUPAC: {predicted_iupac}")
    print(f"Valid Topology: {result.is_valid}")
    
    if not result.is_valid:
        print(f"Error: {result.error}")
        
    print("\nSynthesizing SNFG from prediction...")
    synth = IUPACSynthesizer()
    try:
        synth_tensor = synth.synthesize(predicted_iupac)
    except Exception as e:
        print(f"Failed to synthesize image from prediction: {e}")
        return

    # Convert to PIL for matplotlib
    original_pil = F.to_pil_image(img_tensor)
    synth_pil = F.to_pil_image(synth_tensor)
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Wrap long IUPAC strings for the title
    import textwrap
    wrapped_title = textwrap.fill(predicted_iupac, width=60)
    fig.suptitle(f"OCR Prediction:\n{wrapped_title}", fontsize=14)
    
    axes[0].imshow(original_pil)
    axes[0].set_title("Original Input Image")
    axes[0].axis('off')
    
    axes[1].imshow(synth_pil)
    axes[1].set_title("Reconstructed from Prediction")
    axes[1].axis('off')
    
    color = "green" if result.is_valid else "red"
    valid_text = "Valid Topology" if result.is_valid else f"Invalid: {result.error}"
    fig.text(0.5, 0.05, valid_text, ha='center', fontsize=12, color=color, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    app()
