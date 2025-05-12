export CUDA_VISIBLE_DEVICES=0,1,2,3
# weight_path=outputs/debug/checkpoint-100/diffusion_pytorch_model.safetensors
torchrun \
 --nproc_per_node 4 \
 --master-port 29501 \
 valid_traning.py \
 --task t2v-14B \
 --size 1280*720 \
 --sample_steps 50 \
 --use_original_model \
 --output_dir outputs/exp1_wan_moviidb_720p \
 --ckpt_dir /vepfs-zulution/models/Wan2.1-T2V-14B \
 --dit_fsdp --t5_fsdp --ulysses_size 4 --ring_size 1 --sample_shift 5 \
 --sample_guide_scale 5.0 --base_seed 42 --frame_num 81
