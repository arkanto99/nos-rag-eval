#!/bin/bash

CONFIGS=/home/compartido/pabloF/nos-rag-eval/rag_retriever/configs/experiments

python3 generate_testset.py --config $CONFIGS/bm25.yaml
python3 generate_testset.py --config $CONFIGS/bge-m3.yaml
python3 generate_testset.py --config $CONFIGS/all-minilm-l6-v2.yaml