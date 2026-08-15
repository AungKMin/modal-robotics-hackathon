# Simple Text-to-Image

A simple deployment of the Stable Diffusion 3.5 Large Turbo text-to-image model.

## Model

This template uses [Stable Diffusion 3.5 Large Turbo](https://huggingface.co/adamo1139/stable-diffusion-3.5-large-turbo-ungated), a diffusion model from Stability AI that generates images based on text prompts. Runs on an H100. Inference takes ~1 minute to [cold start](https://modal.com/docs/guide/cold-start) and images are generated in ~1 second.

## Deploy

```bash
modal deploy -m text_to_image.app
```

## Usage

### Programmatic access

```bash
# Sample client script to generate an image (uses a test prompt if omitted)
python text_to_image/try.py "A cat loafing in a sunbeam"
```

### Web

A [web endpoint](https://modal.com/docs/guide/webhook-urls) is automatically generated for this inference function. The URL is printed out when you deploy the app; you can also find it in the Modal web dashboard under the page for the function.

```bash
curl \
  --get '<URL>' \
  --data-urlencode 'prompt=<PROMPT>' \
  --output 'bootstrap-results-text_to_image/generated_image.png'
```
