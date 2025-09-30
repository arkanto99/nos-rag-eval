#!/bin/bash

source /home/compartido/pabloF/load_env.sh

BASE_DIR="/home/compartido/pabloF/nos-rag-eval/results/EN_results"

# Recorre todos los subdirectorios dentro de BASE_DIR
for EXP_DIR in "$BASE_DIR"/*/; do
    echo "Procesando directorio: $EXP_DIR"

    # Ejecutar agregación en cada subdirectorio
    python3 utils/aggregate_metrics.py --folder "$EXP_DIR" --output "aggregate_metrics.jsonl"
done
