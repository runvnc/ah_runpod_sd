# RunPod Stable Diffusion Plugin

A MindRoot plugin that provides text-to-image generation using RunPod's serverless Stable Diffusion endpoints.

## Features
- Text to image generation via RunPod
- Model selection support
- Configurable image parameters
- Base64 image handling

## Configuration
Required environment variables:
- RUNPOD_API_KEY: Your RunPod API key

## Services
- text_to_image: Generate images from text descriptions
- select_image_model: Select appropriate model for image generation

## Commands
- image: Generate an image from a text description

## Installation
```bash
pip install -e .
```

## Usage Example
```python
# Using the service directly
from ah_runpod_sd.mod import text_to_image

image_path = await text_to_image(
    prompt="A beautiful sunset over mountains",
    w=1024,
    h=1024,
    steps=20,
    cfg=8
)

# Or via command
await context.execute_command('image', {
    'description': 'A beautiful sunset over mountains'
})
```
