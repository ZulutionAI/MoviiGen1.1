#!/bin/bash 
 
export PYTHONPATH=./:${PYTHONPATH}
OUTPUT_DIR=YOUR_OUTPUT_DIR
DATA_JSON_PATH=YOUR_DATA_JSON_PATH
torchrun \
    --nnodes $MLP_WORKER_NUM \
    --node_rank $MLP_ROLE_INDEX \
    --nproc_per_node $MLP_WORKER_GPU \
    --master_addr $MLP_WORKER_0_HOST \
    --master_port $MLP_WORKER_0_PORT \
    scripts/train/finetune.py \
    --max_seq_len 170100 \
    --master_weight_type bf16 \
    --ckpt_dir ./MoviiGen1.1 \
    --output_dir ${OUTPUT_DIR} \
    --checkpointing_steps 100 \
    --seed 42 \
    --gradient_checkpointing \
    --data_json_path ${DATA_JSON_PATH} \
    --train_batch_size 1 \
    --num_latent_t 21 \
    --sp_size 8 \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 1 \
    --max_train_steps 1000 \
    --learning_rate 1e-6 \
    --mixed_precision bf16 \
    --checkpoints_total_limit 3 \
    --allow_tf32 \
    --num_height 1080 \
    --num_width 1920 \
    --group_frame \
    --group_resolution \
    --group_ar
