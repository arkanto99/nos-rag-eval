# Nós Rag Evaluation Tool

## Overview
The Nós Rag Evaluation Tool is designed to evaluate retrieval-augmented generation (RAG) systems, focusing on the retrieval and reranking steps. It integrates various components to process user queries, retrieve relevant contexts, and generate responses based on metadata-rich datasets.

## Project Structure
- **datasets/**: Contains datasets used for evaluation.
  - `evaluation_dataset_with_metadata.json`: Stores user queries, reference answers, and retrieved contexts.
  - `qwen_samples_context_fixed.json`: Additional dataset for evaluation.
- **ir-metrics/**: Implements traditional IR metrics for evaluation.
  - `context_entity_recall.py`: Calculates recall based on entity extraction.
  - `test_metrics.py`: Unit tests for IR metrics.
  - `traditional_metrics.py`: Implements precision, recall, and mean reciprocal rank (MRR).
- **llm-as-judge/**: Implements evaluation using the Selene model as a judge.
  - `judge.py`: Defines logic for evaluating Context Precision and Context Recall.
  - `prompts.py`: Contains prompt templates for the Selene model.
- **rag_retriever/**: Implements the RAG system, including context retrieval and reranking logic.
  - `__init__.py`: Initializes the module.
  - `rag_retriever.py`: Defines the `RAG` class and methods for retrieving and formatting contexts.
  - **configs/**: Configuration files for retriever and reranker.
  - **retriever/**: Contains logic for retrieving contexts.
- **utils/**: Utility functions for loading and processing datasets.
  - `dataloader_evaluation.py`: Handles dataset loading and preprocessing.
- **generate_testset.py**: Python script for generating evaluation datasets by processing predefined user queries and retrieving relevant contexts from the Elasticsearch database.
- **README.md**: Documentation for the project.

## Key Features
- **Retrieval Evaluation**: Focuses on evaluating the retrieval and reranking steps of the RAG system.
- **Evaluation Metrics**:
  - **Traditional IR Metrics**: Precision, Recall, and Mean Reciprocal Rank (MRR).
  - **LLM-as-Judge**: Uses the Selene model to compute Context Precision and Context Recall.
- **Dataset Generation**: Automates the creation of datasets for evaluation using predefined questions and Elasticsearch queries.

## Usage
1. **Dataset Preparation**: Ensure the datasets are stored in the `datasets/` directory.
2. **Run Evaluation**: Use `generate_testset.py` to process queries and generate evaluation results.
   ```bash
   python generate_testset.py