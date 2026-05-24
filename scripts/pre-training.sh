export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_ENDPOINT="https://hf-mirror.com"

accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=8 \
  src/pre_training.py \
    