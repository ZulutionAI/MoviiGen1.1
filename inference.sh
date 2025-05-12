export CUDA_VISIBLE_DEVICES=0,1,2,3
# weight_path=outputs/debug/checkpoint-100/diffusion_pytorch_model.safetensors
torchrun \
 --nproc_per_node 4 \
 --master-port 29500 \
 generate.py \
 --task t2v-14B \
 --size 1280*720 \
 --sample_steps 50 \
 --weight_path outputs/debug/checkpoint-10/diffusion_pytorch_model.safetensors \
 --ckpt_dir /vepfs-zulution/models/Wan2.1-T2V-14B \
 --dit_fsdp --t5_fsdp --ulysses_size 4 --ring_size 1 --sample_shift 5\
 --sample_guide_scale 5.0 --prompt "A cat walks on the grass, realistic style." --base_seed 42 --frame_num 81 --save_file outputs/debug/1280*720_step50_shift5_guide5.0.mp4
