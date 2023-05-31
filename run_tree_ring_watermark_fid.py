import argparse
import wandb
import copy
from tqdm import tqdm
import json

import torch

from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from optim_utils import *
from io_utils import *
from pytorch_fid.fid_score import *


def main(args):
    table = None
    if args.with_tracking:
        wandb.init(project='diffusion_watermark', name=args.run_name, tags=['tree_ring_watermark_fid'])
        wandb.config.update(args)
        table = wandb.Table(columns=['gen_no_w', 'gen_w', 'prompt'])
    
    # load diffusion model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder='scheduler')
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        revision='fp16',
        )
    pipe = pipe.to(device)

    # hard coding for now
    with open(args.prompt_file) as f:
        dataset = json.load(f)
        image_files = dataset['images']
        dataset = dataset['annotations']
        prompt_key = 'caption'
    
    no_w_dir = f'fid_outputs/coco/{args.run_name}/no_w_gen'
    w_dir = f'fid_outputs/coco/{args.run_name}/w_gen'
    os.makedirs(no_w_dir, exist_ok=True)
    os.makedirs(w_dir, exist_ok=True)

    # ground-truth patch
    gt_patch = get_watermarking_pattern(pipe, args, device)

    for i in tqdm(range(args.start, args.end)):
        seed = i + args.gen_seed
        
        current_prompt = dataset[i][prompt_key]
        
        ### generation
        # generation without watermarking
        set_random_seed(seed)
        init_latents_no_w = pipe.get_random_latents()

        if args.run_no_w:
            outputs_no_w = pipe(
                current_prompt,
                num_images_per_prompt=args.num_images,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                height=args.image_length,
                width=args.image_length,
                latents=init_latents_no_w,
                )
            orig_image_no_w = outputs_no_w.images[0]
        else:
            orig_image_no_w = None
        
        # generation with watermarking
        if init_latents_no_w is None:
            set_random_seed(seed)
            init_latents_w = pipe.get_random_latents()
        else:
            init_latents_w = copy.deepcopy(init_latents_no_w)

        # get watermarking mask
        watermarking_mask = get_watermarking_mask(init_latents_w, args, device)

        # inject watermark
        init_latents_w = inject_watermark(init_latents_w, watermarking_mask,gt_patch, args)

        outputs_w = pipe(
            current_prompt,
            num_images_per_prompt=args.num_images,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=init_latents_w,
            )
        orig_image_w = outputs_w.images[0]

        if args.with_tracking:
            if i < args.max_num_log_image:
                if args.run_no_w:
                    table.add_data(wandb.Image(orig_image_no_w), wandb.Image(orig_image_w), current_prompt)
                else:
                    table.add_data(None, wandb.Image(orig_image_w), current_prompt)
            else:
                table.add_data(None, None, current_prompt)
        
        image_file_name = image_files[i]['file_name']
        if args.run_no_w:
            orig_image_no_w.save(f'{no_w_dir}/{image_file_name}')
        orig_image_w.save(f'{w_dir}/{image_file_name}')

    ### calculate fid
    try:
        num_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        num_cpus = os.cpu_count()

    num_workers = min(num_cpus, 8) if num_cpus is not None else 0

    # fid for no_w
    if args.run_no_w:
        fid_value_no_w = calculate_fid_given_paths([args.gt_folder, no_w_dir],
                                            50,
                                            device,
                                            2048,
                                            num_workers)
    else:
        fid_value_no_w = None

    # fid for w
    fid_value_w = calculate_fid_given_paths([args.gt_folder, w_dir],
                                          50,
                                          device,
                                          2048,
                                          num_workers)

    if args.with_tracking:
        wandb.log({'Table': table})
        wandb.log({'fid_no_w': fid_value_no_w, 'fid_w': fid_value_w})

    print(f'fid_no_w: {fid_value_no_w}, fid_w: {fid_value_w}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='diffusion watermark')
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=10, type=int)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--with_tracking', action='store_true')
    parser.add_argument('--num_images', default=1, type=int)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--max_num_log_image', default=100, type=int)
    parser.add_argument('--run_no_w', action='store_true')
    parser.add_argument('--gen_seed', default=0, type=int)
    parser.add_argument('--prompt_file', default='fid_outputs/coco/meta_data.json')
    parser.add_argument('--gt_folder', default='fid_outputs/coco/ground_truth')

    # watermark
    parser.add_argument('--w_seed', default=999999, type=int)
    parser.add_argument('--w_channel', default=0, type=int)
    parser.add_argument('--w_pattern', default='rand')
    parser.add_argument('--w_mask_shape', default='circle')
    parser.add_argument('--w_radius', default=10, type=int)
    parser.add_argument('--w_measurement', default='l1_complex')
    parser.add_argument('--w_injection', default='complex')
    parser.add_argument('--w_pattern_const', default=0, type=float)

    args = parser.parse_args()
    
    main(args)