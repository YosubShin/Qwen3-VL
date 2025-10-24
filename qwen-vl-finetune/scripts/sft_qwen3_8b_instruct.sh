#!/bin/bash
set -x

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}

# DeepSpeed configuration
deepspeed=./scripts/zero3.json

# Model configuration
llm=Qwen/Qwen3-VL-8B-Instruct  # Using HuggingFace model ID

# Training hyperparameters
lr=1e-5
batch_size=4
grad_accum_steps=4
eval_batch_size=4

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration (replace with public dataset names)
datasets=m2sv-sft-11k-full

# Output configuration
run_name="qwen3-vl-8b-instruct"
output_dir=./output/qwen3-vl-8b-instruct

# Training arguments
args="
    --deepspeed ${deepspeed}
    --model_name_or_path ${llm}
    --dataset_use ${datasets}
    --data_flatten True
    --tune_mm_vision False
    --tune_mm_mlp True
    --tune_mm_llm True
    --bf16
    --output_dir ${output_dir}
    --num_train_epochs 4
    --per_device_train_batch_size ${batch_size}
    --per_device_eval_batch_size ${eval_batch_size}
    --gradient_accumulation_steps ${grad_accum_steps}
    --max_pixels 50176
    --min_pixels 784
    --eval_strategy steps
    --do_eval
    --eval_steps 40
    --predict_with_generate False
    --prediction_loss_only True
    --eval_accumulation_steps 1
    --save_strategy steps
    --save_steps 40
    --save_total_limit 1
    --early_stopping_patience 3
    --early_stopping_threshold 0.0
    --load_best_model_at_end True
    --metric_for_best_model eval_loss
    --greater_is_better False
    --learning_rate ${lr}
    --weight_decay 0
    --warmup_ratio 0.03
    --max_grad_norm 1
    --lr_scheduler_type cosine
    --logging_steps 1
    --model_max_length 8192
    --gradient_checkpointing True
    --dataloader_num_workers 4
    --run_name ${run_name}
    --report_to wandb"

# Launch training
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}