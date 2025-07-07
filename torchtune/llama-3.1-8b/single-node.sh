#!/bin/bash
LLAMA_MODEL=Llama-3.1-8B-Instruct
checkpoint_dir=$(pwd)/models/$LLAMA_MODEL

tune run \
  lora_finetune_single_device \
  --config llama3_1/8B_qlora_single_device \
  checkpointer.checkpoint_dir=$checkpoint_dir \
  tokenizer.path=$checkpoint_dir/original/tokenizer.model
