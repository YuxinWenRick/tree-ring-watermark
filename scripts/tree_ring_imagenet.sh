### all cases
python run_tree_ring_watermark_imagenet.py --run_name imgnet_no_attack --model_id 256x256_diffusion --w_radius 10 --w_channel 2 --w_pattern ring --start 0 --end 1000 --with_tracking --reference_model dummy
python run_tree_ring_watermark_imagenet.py --run_name imgnet_rotation --model_id 256x256_diffusion --w_radius 10 --w_channel 2 --w_pattern ring --start 0 --end 1000 --r_degree 75 --with_tracking
python run_tree_ring_watermark_imagenet.py --run_name imgnet_jpeg --model_id 256x256_diffusion --w_radius 10 --w_channel 2 --w_pattern ring --start 0 --end 1000 --jpeg_ratio 25 --with_tracking
python run_tree_ring_watermark_imagenet.py --run_name imgnet_cropping --model_id 256x256_diffusion --w_radius 10 --w_channel 2 --w_pattern ring --start 0 --end 1000 --crop_scale 0.75 --crop_ratio 0.75 --with_tracking
python run_tree_ring_watermark_imagenet.py --run_name imgnet_blurring --model_id 256x256_diffusion --w_radius 10 --w_channel 2 --w_pattern ring --start 0 --end 1000 --gaussian_blur_r 4 --with_tracking
python run_tree_ring_watermark_imagenet.py --run_name imgnet_noise --model_id 256x256_diffusion --w_radius 10 --w_channel 2 --w_pattern ring --start 0 --end 1000 --gaussian_std 0.1 --with_tracking
python run_tree_ring_watermark_imagenet.py --run_name imgnet_color_jitter --model_id 256x256_diffusion --w_radius 10 --w_channel 2 --w_pattern ring --start 0 --end 1000 --brightness_factor 6 --with_tracking

### fid
python run_tree_ring_watermark_imagenet_fid.py --run_name imgnet_fid_run --gt_data imagenet --model_id 256x256_diffusion --w_radius 10 --w_channel 2 --w_pattern ring --start 0 --end 10000 --with_tracking
