#!/bin/bash
HF_TOKEN=$HF_TOKEN
LLAMA_MODEL=Llama-3.1-8B-Instruct

tune download meta-llama/$LLAMA_MODEL \
  --hf-token $HF_TOKEN \
  --ignore-patterns "consolidated/*.pth" \
  --output-dir $(pwd)/models/$LLAMA_MODEL
