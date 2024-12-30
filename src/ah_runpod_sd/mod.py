from typing import Optional, Dict, Any, List
import asyncio
import aiohttp
import os
import runpod
from runpod import AsyncioEndpoint, AsyncioJob
from nanoid import generate
from lib.providers.services import service
from lib.providers.commands import command
from PIL import Image
import base64
import io
import traceback


runpod.api_key = os.getenv("RUNPOD_API_KEY")

def random_img_fname() -> str:
    """Generate a random filename for an image with .png extension"""
    return generate() + ".png"

async def send_job(input: Dict[str, Any], endpoint_id: str) -> Optional[Image.Image]:
    """Send a job to RunPod endpoint and wait for results

    Args:
        input: Dictionary of input parameters for the model
        endpoint_id: RunPod endpoint ID

    Returns:
        PIL Image object or None if job fails
    """
    try:
        async with aiohttp.ClientSession() as session:
            endpoint = AsyncioEndpoint(endpoint_id, session)
            job: AsyncioJob = await endpoint.run(input)

            while True:
                status = await job.status()
                print(f"Current job status: {status}")
                
                if status == "COMPLETED":
                    output = await job.output()
                    image_data = output['image_url']
                    
                    # Handle different base64 prefixes
                    for prefix in ['data:image/png;base64,', 'data:image/jpeg;base64,']:
                        if image_data.startswith(prefix):
                            image_data = image_data[len(prefix):]
                            break

                    image_bytes = base64.b64decode(image_data)
                    return Image.open(io.BytesIO(image_bytes)).convert("RGB")
                
                elif status == "FAILED":
                    print("Job failed or encountered an error.")
                    return None
                
                await asyncio.sleep(1)
    except Exception as e:
        print(f"Error in send_job: {str(e)}")
        return None

@service()
async def select_image_model(context: Optional[Any] = None, model_id: Optional[str] = None, 
                           local: bool = False, uncensored: bool = False) -> Dict[str, Any]:
    """Select an appropriate image model based on criteria

    Args:
        context: Context object containing user preferences
        model_id: Specific model ID to select
        local: Whether to use local models only
        uncensored: Whether to allow uncensored models

    Returns:
        Dictionary containing model information
    """
    models = await select_models(
        service_or_command='image',
        provider='AH Runpod',
        local=False,
        model_id=model_id,
        uncensored=context.uncensored if context else False
    )
    return models[0]

async def default_image_model(context):
    return {
        'endpoint_id':  os.getenv("RUNPOD_SD_ENDPOINT_ID"),
        'defaults': {
            'steps': 25,
            'cfg': 8.0,
            'prompt': '',
            'negative_prompt': 'ugly, old, fat, bizarre, low quality, score_3, score_4'
        }
    }


@service()
async def text_to_image(prompt: str, negative_prompt: str = '', model_id: Optional[str] = None,
                       from_huggingface: Optional[str] = None, count: int = 1,
                       context: Optional[Any] = None, save_to: Optional[str] = None,
                       w: int = 1024, h: int = 1024, steps: int = 20, cfg: float = 8.0) -> Optional[str]:
    """Generate image(s) from text description

    Args:
        prompt: Text description of desired image
        negative_prompt: Things to avoid in the image
        model_id: Specific model to use
        from_huggingface: HuggingFace model ID
        count: Number of images to generate
        context: Context object
        save_to: Path to save image
        w: Image width
        h: Image height
        steps: Number of inference steps
        cfg: Guidance scale

    Returns:
        Path to saved image or None if generation fails
    """
    try:
        model = await default_image_model(context)
        print(f"Using model: {model}")

        endpoint_id = model['endpoint_id']
        
        input_params = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": steps,
            "refiner_inference_steps": 0,
            "width": w,
            "height": h,
            "guidance_scale": cfg,
            "strength": 0.3,
            "seed": None,
            "num_images": 1
        }

        # Apply model defaults if available
        if 'defaults' in model:
            defaults = model['defaults']
            mapping = {
                'steps': 'num_inference_steps',
                'cfg': 'guidance_scale',
                'seed': 'seed',
                'prompt': 'prompt',
                'negative_prompt': 'negative_prompt',
                'width': 'width',
                'height': 'height'
            }
            
            for model_key, input_key in mapping.items():
                if model_key in defaults:
                    if model_key in ['prompt', 'negative_prompt']:
                        input_params[input_key] += ',' + defaults[model_key]
                    else:
                        input_params[input_key] = defaults[model_key]

        for n in range(count):
            print(f"Generating image {n+1}/{count}")
            image = await send_job(input_params, endpoint_id)
            
            if image:
                fname = save_to or os.path.join("imgs", random_img_fname())
                os.makedirs(os.path.dirname(fname), exist_ok=True)
                image.save(fname)
                return fname
            
        return None

    except Exception as e:
        trace = traceback.format_exc()
        print(f"Error in text_to_image: {str(e)} \n {trace}")
        return None

@command()
async def image(description: str = "", context: Optional[Any] = None) -> None:
    """Generate an image from a text description

    Args:
        description: Text description of desired image
        context: Context object

    Example:
        [
          { "image": {"description": "A cute tabby cat in the forest"} },
          { "image": {"description": "A happy golden retriever in the park"} }
        ]
    """
    try:
        fname = await context.text_to_image(description)
        if fname:
            print(f"Image saved to: {fname}")
            await context.insert_image(fname)
        else:
            print("Failed to generate image")
    except Exception as e:
        print(f"Error in image command: {str(e)}")
