export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun \
 --nproc_per_node 4 \
 --master-port 29500 \
 generate.py \
 --task t2v-14B \
 --size 1280*720 \
 --sample_steps 50 \
 --ckpt_dir ./MoviiGen1.1 \
 --dit_fsdp --t5_fsdp --ulysses_size 4 --ring_size 1 --sample_shift 5\
 --sample_guide_scale 5.0 --prompt "A cat walks on the grass, realistic style." --base_seed 42 --frame_num 81
