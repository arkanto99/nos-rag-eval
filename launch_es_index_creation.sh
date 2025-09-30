#!/bin/bash


 python3 elasticsearch/create_elastic_database_with_embeddings.py \
     --index "/home/compartido/pabloF/nos-rag-eval/elasticsearch/index_embedding_example.json" \
     --hf_cache_dir "/home/compartido/pabloF/cache" \
     --data_path "/home/compartido/pabloF/nos-rag-eval/datasets/News" \
     --chunking "paragraph" \
     --embedding "Qwen/Qwen3-Embedding-0.6B"
