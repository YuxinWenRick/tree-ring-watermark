import torch
from typing import Union, List, Tuple
import numpy as np
import hashlib
import os
import tempfile
from huggingface_hub import hf_api
from .utils import get_org

api = hf_api.HfApi()

def _circle_mask(size=64, r=10, x_offset=0, y_offset=0):
    # reference: https://stackoverflow.com/questions/69687798/generating-a-soft-circluar-mask-using-numpy-python-3
    x0 = y0 = size // 2
    x0 += x_offset
    y0 += y_offset
    y, x = np.ogrid[:size, :size]
    y = y[::-1]

    return ((x - x0)**2 + (y-y0)**2)<= r**2

def _get_pattern(shape, w_pattern='ring', generator=None):
    gt_init = torch.randn(shape, generator=generator)

    if 'rand' in w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
        gt_patch[:] = gt_patch[0]
    elif 'zeros' in w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
    elif 'ring' in w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))

        gt_patch_tmp = gt_patch.clone().detach()
        for i in range(shape[-1] // 2, 0, -1):
            tmp_mask = _circle_mask(gt_init.shape[-1], r=i)
            tmp_mask = torch.tensor(tmp_mask)
            
            for j in range(gt_patch.shape[1]):
                gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()

    return gt_patch


# def get_noise(shape: Union[torch.Size, List, Tuple], model_hash: str) -> torch.Tensor:
def get_noise(shape: Union[torch.Size, List, Tuple], model_hash: str, generator=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:    
    # for now we hard code all hyperparameters
    w_channel = 0 # id for watermarked channel
    w_radius = 10 # watermark radius
    w_pattern = 'rand' # watermark pattern

    # get watermark key and mask
    np_mask = _circle_mask(shape[-1], r=w_radius)
    torch_mask = torch.tensor(np_mask)
    w_mask = torch.zeros(shape, dtype=torch.bool)
    w_mask[:, w_channel] = torch_mask
    
    w_key = _get_pattern(shape, w_pattern=w_pattern, generator=generator)

    # inject watermark
    assert len(shape) == 4, f"Make sure you pass a `shape` tuple/list of length 4 not {len(shape)}"
    assert shape[0] == 1, f"For now only batch_size=1 is supported, not {shape[0]}."

    init_latents = torch.randn(shape, generator=generator)

    init_latents_fft = torch.fft.fftshift(torch.fft.fft2(init_latents), dim=(-1, -2))
    init_latents_fft[w_mask] = w_key[w_mask].clone()
    init_latents = torch.fft.ifft2(torch.fft.ifftshift(init_latents_fft, dim=(-1, -2))).real

    # convert the tensor to bytes
    tensor_bytes = init_latents.numpy().tobytes()

    # generate a hash from the bytes
    hash_object = hashlib.sha256(tensor_bytes)
    hex_dig = hash_object.hexdigest()

    file_name = "_".join([hex_dig, str(w_channel), str(w_radius), w_pattern]) + ".npy"
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, file_name)
    np.save(file_path, w_key)

    org = get_org()
    repo_id = os.path.join(org, model_hash)

    api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="dataset")

    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_name,
        repo_id=repo_id,
        repo_type="dataset",
    )

    return init_latents
