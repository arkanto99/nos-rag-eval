#!/bin/bash

CONFIGS=/home/compartido/pabloF/nos-rag-eval/rag_retriever/configs/experiments

BM25=$CONFIGS/bm25.yaml
BM25_BGE_RERANKER=$CONFIGS/bm25_bge-reranker.yaml
BGE_M3=$CONFIGS/bge-m3.yaml
BGE_M3_BGE_RERANKER=$CONFIGS/bge-m3_bge-reranker.yaml
MINILM=$CONFIGS/all-minilm-l6-v2.yaml
MINILM_BGE_RERANKER=$CONFIGS/all-minilm-l6-v2_bge-reranker.yaml

python3 generate_testset.py --config $MINILM
python3 generate_testset.py --config $MINILM_BGE_RERANKER