from huggingface_hub import snapshot_download
import numpy as np
import torch
from torchvision import transforms
import PIL
from typing import Union
from huggingface_hub import snapshot_download
from diffusers import DDIMInverseScheduler
from .utils import get_org
from ._get_noise import _circle_mask
import os

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


def load_keys(cache_dir):
    # Initialize an empty dictionary to store the numpy arrays
    arrays = {}

    # List all files in the directory
    for file_name in os.listdir(cache_dir):
        # Check if the file is a .npy file
        if file_name.endswith('.npy'):
            # Define the file path
            file_path = os.path.join(cache_dir, file_name)

            # Load the numpy array and store it in the dictionary
            arrays[file_name] = np.load(file_path)

    # Return the 'arrays' dictionary
    return arrays


# def detect(image: Union[PIL.Image.Image, torch.Tensor, np.ndarray], model_hash: str):
def detect(image: Union[PIL.Image.Image, torch.Tensor, np.ndarray], pipe, model_hash):
    detection_time_num_inference = 50
    threshold = 77

    org = get_org()
    repo_id = os.path.join(org, model_hash)

    cache_dir = snapshot_download(repo_id, repo_type="dataset")
    keys = load_keys(cache_dir)

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
    inverted_latents = inverted_latents.images.float().cpu()

    # check if one key matches
    shape = image_latents.shape
    for filename, w_key in keys.items():
        w_channel, w_radius = filename.split(".npy")[0].split("_")[1:3]

        np_mask = _circle_mask(shape[-1], r=int(w_radius))
        torch_mask = torch.tensor(np_mask)
        w_mask = torch.zeros(shape, dtype=torch.bool)
        w_mask[:, int(w_channel)] = torch_mask

        # calculate the distance
        inverted_latents_fft = torch.fft.fftshift(torch.fft.fft2(inverted_latents), dim=(-1, -2))
        dist = torch.abs(inverted_latents_fft[w_mask] - w_key[w_mask]).mean().item()

        if dist <= threshold:
            pipe.scheduler = curr_scheduler
            return True

    return False
