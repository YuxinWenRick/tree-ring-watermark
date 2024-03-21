import argparse
import wandb
import copy
from tqdm import tqdm
from statistics import mean, stdev
from sklearn import metrics

import torch

from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

from optim_utils import *
from io_utils import *
from pytorch_fid.fid_score import *


def main(args):
    table = None
    if args.with_tracking:
        wandb.init(
            project="diffusion_watermark",
            name=args.run_name,
            tags=["tree_ring_watermark_imagenet_fid"],
        )
        wandb.config.update(args)
        table = wandb.Table(columns=["gen_no_w", "gen_w"])

    # load diffusion model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args.timestep_respacing = f"ddim{args.num_inference_steps}"

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    shape = (args.bs, 3, args.image_size, args.image_size)

    no_w_dir = f"fid_outputs/{args.gt_data}/{args.run_name}/no_w_gen"
    w_dir = f"fid_outputs/{args.gt_data}/{args.run_name}/w_gen"
    os.makedirs(no_w_dir, exist_ok=True)
    os.makedirs(w_dir, exist_ok=True)

    # ground-truth patch
    gt_patch = get_watermarking_pattern(None, args, device, shape=shape)

    num_iters = (args.end - args.start) // args.bs
    counter = 0

    for i in tqdm(range(num_iters)):
        seed = i + args.gen_seed

        ### generation
        # generation without watermarking
        set_random_seed(seed)
        init_latents_no_w = torch.randn(*shape, device=device)

        model_kwargs = {}
        if args.class_cond:
            classes = torch.randint(
                low=0, high=NUM_CLASSES, size=(args.bs,), device=device
            )
            model_kwargs["y"] = classes

        if args.run_no_w:
            outputs_no_w = diffusion.ddim_sample_loop(
                model=model,
                shape=shape,
                noise=init_latents_no_w,
                model_kwargs=model_kwargs,
                device=device,
                return_image=True,
            )
            orig_image_no_ws = outputs_no_w
        else:
            orig_image_no_ws = None

        # generation with watermarking
        if init_latents_no_w is None:
            set_random_seed(seed)
            init_latents_w = torch.randn(*shape, device=device)
        else:
            init_latents_w = copy.deepcopy(init_latents_no_w)

        # get watermarking mask
        watermarking_mask = get_watermarking_mask(init_latents_w, args, device)

        # inject watermark
        init_latents_w = inject_watermark(
            init_latents_w, watermarking_mask, gt_patch, args
        )

        outputs_w = diffusion.ddim_sample_loop(
            model=model,
            shape=shape,
            noise=init_latents_w,
            model_kwargs=model_kwargs,
            device=device,
            return_image=True,
        )
        orig_image_ws = outputs_w

        for j in range(len(orig_image_ws)):
            if orig_image_no_ws is None:
                orig_image_no_w = None
            else:
                orig_image_no_w = orig_image_no_ws[j]

            orig_image_w = orig_image_ws[j]

            if args.with_tracking:
                if counter < args.max_num_log_image:
                    if args.run_no_w:
                        table.add_data(
                            wandb.Image(orig_image_no_w), wandb.Image(orig_image_w)
                        )
                    else:
                        table.add_data(None, wandb.Image(orig_image_w))
                else:
                    table.add_data(None, None)

            image_file_name = f"{counter}.jpg"
            if args.run_no_w:
                orig_image_no_w.save(f"{no_w_dir}/{image_file_name}")
            orig_image_w.save(f"{w_dir}/{image_file_name}")

            counter += 1

    ### calculate fid
    try:
        num_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        num_cpus = os.cpu_count()

    num_workers = min(num_cpus, 8) if num_cpus is not None else 0

    # fid for no_w
    if args.run_no_w:
        fid_value_no_w = calculate_fid_given_paths(
            [f"fid_outputs/{args.gt_data}/ground_truth", no_w_dir],
            50,
            device,
            2048,
            num_workers,
        )
    else:
        fid_value_no_w = None

    # fid for w
    fid_value_w = calculate_fid_given_paths(
        [f"fid_outputs/{args.gt_data}/ground_truth", w_dir],
        50,
        device,
        2048,
        num_workers,
    )

    if args.with_tracking:
        wandb.log({"Table": table})
        wandb.log({"fid_no_w": fid_value_no_w, "fid_w": fid_value_w})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="diffusion watermark")
    parser.add_argument("--run_name", default="test")
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=10, type=int)
    parser.add_argument("--image_length", default=512, type=int)
    parser.add_argument("--model_id", default="256x256_diffusion")
    parser.add_argument("--gt_data", default="imagenet")
    parser.add_argument("--with_tracking", action="store_true")
    parser.add_argument("--num_images", default=1, type=int)
    parser.add_argument("--guidance_scale", default=7.5, type=float)
    parser.add_argument("--num_inference_steps", default=50, type=int)
    parser.add_argument("--max_num_log_image", default=100, type=int)
    parser.add_argument("--run_no_w", action="store_true")
    parser.add_argument("--gen_seed", default=0, type=int)
    parser.add_argument("--bs", default=4, type=int)

    # watermark
    parser.add_argument("--w_seed", default=999999, type=int)
    parser.add_argument("--w_channel", default=0, type=int)
    parser.add_argument("--w_pattern", default="rand")
    parser.add_argument("--w_mask_shape", default="circle")
    parser.add_argument("--w_radius", default=10, type=int)
    parser.add_argument("--w_measurement", default="l1_complex")
    parser.add_argument("--w_injection", default="complex")
    parser.add_argument("--w_pattern_const", default=0, type=float)

    args = parser.parse_args()

    args.__dict__.update(model_and_diffusion_defaults())
    args.__dict__.update(read_json(f"openai_config/{args.model_id}.json"))

    main(args)
