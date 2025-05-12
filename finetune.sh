#!/bin/bash 
 
# PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

export PYTHONPATH=./:${PYTHONPATH}


# --master_weight_type bf16\
#--data_json_path data/moviidb_v0.1/preprocess/720p/videos2caption.json \
#--data_json_path /vepfs-zulution/zhangpengpeng/cv/video_generation/Wan2.1/data/mixkit/processed/videos2caption.json \

NCCL_DEBUG=ERROR CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes 1 --nproc_per_node 4 --master-port 29500 scripts/train/finetune.py \
    --max_seq_len 75600 \
    --master_weight_type bf16 \
    --ckpt_dir /cv/bjzhu/models/Wan2.1-T2V-14B \
    --output_dir ${DEFAULT_LOG}/outputs/exp1_wan_moviidb_720p \
    --checkpointing_steps 100 \
    --seed 42 \
    --gradient_checkpointing \
    --data_json_path data/moviidb_v0.1/preprocess/720p/videos2caption.json \
    --train_batch_size 1 \
    --num_latent_t 21 \
    --sp_size 4 \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 1 \
    --max_train_steps 1000 \
    --learning_rate 1e-6 \
    --mixed_precision bf16 \
    --checkpoints_total_limit 3 \
    --allow_tf32 \
    --num_height 720 \
    --num_width 1280 \
    --group_frame \
    --group_resolution
