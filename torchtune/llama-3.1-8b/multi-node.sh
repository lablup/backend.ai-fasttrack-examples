#!/bin/bash
LLAMA_MODEL=Llama-3.1-8B-Instruct
checkpoint_dir=$(pwd)/models/$LLAMA_MODEL

nnodes=$BACKENDAI_CLUSTER_SIZE
nproc_per_node=$(nvidia-smi -L | wc -l)

NODE_RANK=$BACKENDAI_CLUSTER_LOCAL_RANK

tune run \
  --nnodes $nnodes \
  --nproc_per_node $nproc_per_node \
  --node-rank $NODE_RANK \
  --rdzv-id $BACKENDAI_SESSION_ID \
  --rdzv-endpoint main1:12345 \
  --master-addr main1 \
  --master-port 12345 \
  lora_finetune_distributed \
  --config llama3_1/8B_lora \
  lora_attn_modules=['q_proj','k_proj','v_proj','output_proj'] \
  lora_rank=32 lora_alpha=64 output_dir=./lora_experiment_1 \
  checkpointer.checkpoint_dir=$checkpoint_dir \
  tokenizer.path=$checkpoint_dir/original/tokenizer.model
