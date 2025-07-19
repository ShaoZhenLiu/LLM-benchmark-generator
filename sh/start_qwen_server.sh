set -x

# 设置gpu数
export CUDA_VISIBLE_DEVICES=0,1,2,3
# 自动获取GPU数量
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

MODEL_PATH="/data/shaozhen.liu/python_project/hf_models/Qwen2.5-VL-72B-Instruct"
MODEL_NAME="qwen2.5-vl-72b-instruct"

vllm serve $MODEL_PATH \
    --port 18901 \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.9 \
    --tensor-parallel-size $NUM_GPUS \
    --dtype bfloat16 \
    --limit-mm-per-prompt "image=10" \
    --served-model-name $MODEL_NAME \
    --trust-remote-code