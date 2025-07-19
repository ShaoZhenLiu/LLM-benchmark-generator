set -x

python generate_data.py \
    --data_path "./data/emb_cluster_vis_agglomerative.jsonl" \
    --batch_size 16 \
    --output_path "./data" \
