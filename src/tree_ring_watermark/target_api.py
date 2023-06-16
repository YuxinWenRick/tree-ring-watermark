#!/usr/bin/env python3
import torch
import torchvision

from diffusers import DPMSolverMultistepScheduler, DiffusionPipeline, DDIMScheduler
from inverse_stable_diffusion import InversableStableDiffusionPipeline

from _get_noise import get_noise
from _detect import detect

from PIL import Image
import requests
from io import BytesIO

model_id = 'stabilityai/stable-diffusion-2-1-base'

# load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# IMPORTANT: We need to make sure to be able to use a normal diffusion pipeline so that people see 
# the tree-ring-watermark method as general enough
scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder='scheduler')
# or
scheduler = DDIMScheduler.from_pretrained(model_id, subfolder='scheduler')
pipe = DiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
pipe = pipe.to(device)

shape = (1, 4, 64, 64)
latents, w_key, w_mask = get_noise(shape, pipe)

watermarked_image = pipe(prompt="an astronaut", latents=latents).images[0]

is_watermarked = detect(watermarked_image, pipe, w_key, w_mask)
print(f'is_watermarked: {is_watermarked}')
