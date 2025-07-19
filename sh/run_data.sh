set -x

python data.py \
    --data_path "./data/mscoco.parquet" \
    --cache_path "./data/image_cache" \
    --sample_num 10000 \