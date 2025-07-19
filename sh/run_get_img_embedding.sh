set -x

export CUDA_VISIBLE_DEVICES=3

python get_img_embedding.py \
    --data_path "./data/mscoco.parquet" \
    --cache_path "./data/image_cache" \
    --output_path "./data/mscoco_emb.parquet" \
    --sample_num 10000 \
    --model_path "/data/shaozhen.liu/python_project/hf_models/gme-Qwen2-VL-7B-Instruct" \
