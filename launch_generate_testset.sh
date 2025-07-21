#!/bin/bash

CONFIGS=/home/compartido/pabloF/nos-rag-eval/rag_retriever/configs/experiments

BM25=$CONFIGS/bm25.yaml
BM25_BGE_RERANKER=$CONFIGS/bm25_bge-reranker.yaml
BM25_CHUNKED=$CONFIGS/chunking/bm25_chunked.yaml
BM25_BGE_RERANKER_CHUNKED=$CONFIGS/chunking/bm25_bge-reranker_chunked.yaml

BGE_M3=$CONFIGS/bge-m3.yaml
BGE_M3_BGE_RERANKER=$CONFIGS/bge-m3_bge-reranker.yaml
BGE_M3_CHUNKED=$CONFIGS/chunking/bge-m3_chunked.yaml
BGE_M3_BGE_RERANKER_CHUNKED=$CONFIGS/chunking/bge-m3_bge-reranker_chunked.yaml

MINILM=$CONFIGS/all-minilm-l6-v2.yaml
MINILM_BGE_RERANKER=$CONFIGS/all-minilm-l6-v2_bge-reranker.yaml
MINILM_CHUNKED=$CONFIGS/chunking/all-minilm-l6-v2_chunked.yaml
MINILM_BGE_RERANKER_CHUNKED=$CONFIGS/chunking/all-minilm-l6-v2_bge-reranker_chunked.yaml

python3 generate_testset.py --config $BM25_BGE_RERANKER_CHUNKED
#python3 generate_testset.py --config $BGE_M3_BGE_RERANKER_CHUNKED