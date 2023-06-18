import numpy as np
import torch
from torchvision import transforms
import PIL
from typing import Union
from diffusers import DDIMInverseScheduler

def _transform_img(image, target_size=512):
    tform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ]
    )
    image = tform(image)
    return 2.0 * image - 1.0

# def detect(image: Union[PIL.Image.Image, torch.Tensor, np.ndarray], model_hash: str):
def detect(image: Union[PIL.Image.Image, torch.Tensor, np.ndarray], pipe, w_key, w_mask):
    detection_time_num_inference = 50
    threshold = 77

    # ddim inversion
    curr_scheduler = pipe.scheduler
    pipe.scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    img = _transform_img(image).unsqueeze(0).to(pipe.unet.dtype).to(pipe.device)
    image_latents = pipe.vae.encode(img).latent_dist.mode() * 0.18215
    inverted_latents = pipe(
            prompt='',
            latents=image_latents,
            guidance_scale=1,
            num_inference_steps=detection_time_num_inference,
            output_type='latent',
        )
    inverted_latents = inverted_latents.images

    # calculate the distance
    inverted_latents_fft = torch.fft.fftshift(torch.fft.fft2(inverted_latents), dim=(-1, -2))
    dist = torch.abs(inverted_latents_fft[w_mask] - w_key[w_mask]).mean().item()

    pipe.scheduler = curr_scheduler

    if dist <= threshold:
        return True
    else:
        return False
