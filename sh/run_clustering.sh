set -x

python clustering.py \
    --data_path "./data/mscoco_emb.parquet" \
    --cluster_method "agglomerative" \
    --output_path "./data/emb_cluster_vis_agglomerative.jsonl" \
    --n_clusters 15 \
    --sample_per_cluster 10 \
    
