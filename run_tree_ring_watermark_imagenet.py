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


def main(args):
    table = None
    if args.with_tracking:
        wandb.init(project='diffusion_watermark', name=args.run_name, tags=['latent_watermark_fourier_openai'])
        wandb.config.update(args)
        table = wandb.Table(columns=['gen_no_w', 'gen_w', 'no_w_metric', 'w_metric'])

    # load diffusion model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    args.timestep_respacing = f"ddim{args.num_inference_steps}"

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    shape = (args.num_images, 3, args.image_size, args.image_size)

    # ground-truth patch
    gt_patch = get_watermarking_pattern(None, args, device, shape)

    results = []
    no_w_metrics = []
    w_metrics = []

    for i in tqdm(range(args.start, args.end)):
        seed = i + args.gen_seed
        
        ### generation
        model_kwargs = {}
        if args.class_cond:
            classes = torch.randint(
                low=0, high=NUM_CLASSES, size=(args.num_images,), device=device
            )
            model_kwargs["y"] = classes
            
        # generation without watermarking
        set_random_seed(seed)
        init_latents_no_w = torch.randn(*shape, device=device)
        outputs_no_w = diffusion.ddim_sample_loop(
                    model=model,
                    shape=shape,
                    noise=init_latents_no_w,
                    model_kwargs=model_kwargs,
                    device=device,
                    return_image=True,
                )
        orig_image_no_w = outputs_no_w[0]
        
        # generation with watermarking
        if init_latents_no_w is None:
            set_random_seed(seed)
            init_latents_w = torch.randn(*shape, device=device)
        else:
            init_latents_w = copy.deepcopy(init_latents_no_w)

        # get watermarking mask
        watermarking_mask = get_watermarking_mask(init_latents_w, args, device)

        # inject watermark
        init_latents_w = inject_watermark(init_latents_w, watermarking_mask, gt_patch, args)

        outputs_w = diffusion.ddim_sample_loop(
                    model=model,
                    shape=shape,
                    noise=init_latents_w,
                    model_kwargs=model_kwargs,
                    device=device,
                    return_image=True,
                )
        orig_image_w = outputs_w[0]

        ### test watermark
        # distortion
        orig_image_no_w_auged, orig_image_w_auged = image_distortion(orig_image_no_w, orig_image_w, seed, args)

        # reverse img without watermarking
        reversed_latents_no_w = diffusion.ddim_reverse_sample_loop(
                model=model,
                shape=shape,
                image=orig_image_no_w_auged,
                model_kwargs=model_kwargs,
                device=device,
            )

        # reverse img with watermarking
        reversed_latents_w = diffusion.ddim_reverse_sample_loop(
                model=model,
                shape=shape,
                image=orig_image_w_auged,
                model_kwargs=model_kwargs,
                device=device,
            )

        # eval
        no_w_metric, w_metric = eval_watermark(reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args)

        results.append({
            'no_w_metric': no_w_metric, 'w_metric': w_metric,
        })

        no_w_metrics.append(-no_w_metric)
        w_metrics.append(-w_metric)

        if args.with_tracking:
            if (args.reference_model is not None) and (i < args.max_num_log_image):
                # log images when we use reference_model
                table.add_data(wandb.Image(orig_image_no_w), wandb.Image(orig_image_w), no_w_metric, w_metric)
            else:
                table.add_data(None, None, no_w_metric, w_metric)

    # roc
    preds = no_w_metrics +  w_metrics
    t_labels = [0] * len(no_w_metrics) + [1] * len(w_metrics)

    fpr, tpr, thresholds = metrics.roc_curve(t_labels, preds, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    acc = np.max(1 - (fpr + (1 - tpr))/2)
    low = tpr[np.where(fpr<.01)[0][-1]]

    if args.with_tracking:
        wandb.log({'Table': table})
        wandb.log({'auc': auc, 'acc':acc, 'TPR@1%FPR': low})
        
    print(f'auc: {auc}, acc: {acc}, TPR@1%FPR: {low}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='diffusion watermark')
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--dataset', default='Gustavosta/Stable-Diffusion-Prompts')
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=10, type=int)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--model_id', default='256x256_diffusion')
    parser.add_argument('--with_tracking', action='store_true')
    parser.add_argument('--num_images', default=1, type=int)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--test_num_inference_steps', default=None, type=int)
    parser.add_argument('--reference_model', default=None)
    parser.add_argument('--reference_model_pretrain', default=None)
    parser.add_argument('--max_num_log_image', default=100, type=int)
    parser.add_argument('--gen_seed', default=0, type=int)

    # watermark
    parser.add_argument('--w_seed', default=999999, type=int)
    parser.add_argument('--w_channel', default=0, type=int)
    parser.add_argument('--w_pattern', default='rand')
    parser.add_argument('--w_mask_shape', default='circle')
    parser.add_argument('--w_radius', default=10, type=int)
    parser.add_argument('--w_measurement', default='l1_complex')
    parser.add_argument('--w_injection', default='complex')
    parser.add_argument('--w_pattern_const', default=0, type=float)
    
    # for image distortion
    parser.add_argument('--r_degree', default=None, type=float)
    parser.add_argument('--jpeg_ratio', default=None, type=int)
    parser.add_argument('--crop_scale', default=None, type=float)
    parser.add_argument('--crop_ratio', default=None, type=float)
    parser.add_argument('--gaussian_blur_r', default=None, type=int)
    parser.add_argument('--gaussian_std', default=None, type=float)
    parser.add_argument('--brightness_factor', default=None, type=float)
    parser.add_argument('--rand_aug', default=0, type=int)

    args = parser.parse_args()

    args.__dict__.update(model_and_diffusion_defaults())
    args.__dict__.update(read_json(f'{args.model_id}.json'))

    if args.test_num_inference_steps is None:
        args.test_num_inference_steps = args.num_inference_steps
    
    main(args)