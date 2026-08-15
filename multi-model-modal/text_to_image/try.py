import time
from pathlib import Path
from rich import print
from rich.console import Console

import modal

if __name__ == "__main__":
    """
    Generate an image using the deployed Modal function via `ImageGenerator`.

    Args:
        prompt (str): The text prompt to generate an image from.
        results_dir (str): Directory to save the generated image.
        image_name (str): Name of the generated image file (without extension).

    Example:
        python text_to_image.py
        # or with overrides:
        python text_to_image.py "A serene mountain landscape" "./outputs" "mountain_img"
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate an image from a text prompt."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="A princess riding on a pony",
        help="Text prompt to generate an image from.",
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        default=str(Path(__file__).parent),
        help="Directory to save the generated image.",
    )
    parser.add_argument(
        "image_name",
        nargs="?",
        default="generated_image",
        help="Name for the generated image file (without extension).",
    )
    args = parser.parse_args()

    if args.prompt is None:
        print("No prompt provided. Using default prompt.")
        prompt = "A princess riding on a pony"
    else:
        prompt = args.prompt

    prompt = args.prompt
    results_dir = args.results_dir
    image_name = args.image_name

    generator = modal.Cls.from_name("bootstrap-text-to-image", "ImageGenerator")()

    print("\n\n")
    print("--------------------------------------------------------------------------------")
    print(f"Running {Path(__file__).name} to invoke the deployed function")
    print("--------------------------------------------------------------------------------")

    start = time.perf_counter()
    console = Console()
    with console.status(
        (
            "[green]Loading Stable Diffusion 3.5 Large Turbo on a cloud GPU and running inference.[/green]\n"
            f"[green]Prompt: {prompt}[/green]\n"
            f"[green]View progress in Modal dashboard: [magenta]{generator.generate.get_dashboard_url()}[/magenta][/green]"
        ),
        spinner="dots",
    ):
        image_bytes = generator.generate.remote(prompt, batch_size=1)
    elapsed = time.perf_counter() - start

    print(f"[green]Elapsed: {elapsed:.2f} seconds[/green]")
    output_dir = Path(results_dir)
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{image_name}.png"
    output_path.write_bytes(image_bytes[0])

    print(f"[green]✓ Image saved to: {output_path}[/green]")
    print(f"[green]\nGenerate another image by running: python text_to_image/{Path(__file__).name} \"<PROMPT>\"[/green]")
