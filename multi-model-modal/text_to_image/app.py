"""
Generate images from text using Stable Diffusion 3.5 Large Turbo.
"""

import io
import random
from typing import Optional

import modal

app = modal.App(name="bootstrap-text-to-image")

# Configure the container image with necessary dependencies
CACHE_DIR = "/cache"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "accelerate==0.33.0",
        "diffusers==0.31.0",
        "fastapi[standard]==0.115.4",
        "huggingface-hub==0.36.0",
        "sentencepiece==0.2.0",
        "torch==2.5.1",
        "torchvision==0.20.1",
        "transformers~=4.44.0",
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HUB_CACHE": CACHE_DIR,
        }
    )
)

with image.imports():
    import diffusers
    import torch

    from fastapi import Response

MODEL_ID = "adamo1139/stable-diffusion-3.5-large-turbo-ungated"
MODEL_REVISION_ID = "9ad870ac0b0e5e48ced156bb02f85d324b7275d2"

cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

@app.cls(
    image=image,
    gpu="H100",
    timeout=600,
    volumes={CACHE_DIR: cache_volume},
)
class ImageGenerator:
    @modal.enter()
    def load_model(self):
        """Load the model into GPU memory on container startup."""
        self.pipe = diffusers.StableDiffusion3Pipeline.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION_ID,
            torch_dtype=torch.bfloat16,
        ).to("cuda")

    @modal.method()
    def generate(
        self, prompt: str, batch_size: int = 1, seed: Optional[int] = None
    ) -> list[bytes]:
        """Generate images from a text prompt."""
        seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        print(f"Generating image with seed: {seed}")

        torch.manual_seed(seed)
        images = self.pipe(
            prompt,
            num_images_per_prompt=batch_size,
            num_inference_steps=4,
            guidance_scale=0.0,
            max_sequence_length=512,
        ).images

        # Convert images to bytes
        image_output = []
        for image in images:
            with io.BytesIO() as buf:
                image.save(buf, format="PNG")
                image_output.append(buf.getvalue())

        torch.cuda.empty_cache()
        return image_output

    @modal.fastapi_endpoint(docs=True)
    def web(self, prompt: str, seed: Optional[int] = None):
        """
        Web endpoint for generating images via HTTP.
        Visit the /docs endpoint for interactive API documentation.
        """
        return Response(
            content=self.generate.local(prompt, batch_size=1, seed=seed)[0],
            media_type="image/png",
        )
