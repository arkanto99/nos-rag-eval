# Code from https://discuss.huggingface.co/t/use-ragas-with-huggingface-llm/75769

import sys
import os
import json
from tqdm import tqdm

from rag_retriever.rag_retriever import RAG
from utils.dataloader_evaluation import load_qa_with_metadata

dataset = []
dataset = load_qa_with_metadata(file_path="/home/compartido/pabloF/nos-rag-eval/datasets/qwen_samples_context_fixed_HumanRevised.json")
rag = RAG()

# Initialize or load existing results
output_file = 'retrieved_dataset_with_metadata.json'
if os.path.exists(output_file):
    with open(output_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    # Get the last processed id
    processed_ids = {item['id'] for item in results}
else:
    results = []
    processed_ids = set()

try:
    for item in tqdm(dataset, desc="Generating dataset"):
        idx = item['id']
        if idx in processed_ids:
            continue    
        query = item['question']

        try:
            relevant_docs, _ = rag.retrieve_contexts(query)
            retrieved_contexts = []
            for doc in relevant_docs:
                metadata = doc["metadata"]["_source"]
                retrieved_contexts.append({
                    "context": doc["content"],
                    "score": doc["score"],
                    "context_metadata": {
                        "id": metadata["id"],
                        "source_id": metadata.get("source_id",f"Praza-{metadata.get('published_on')}"),
                        "title": metadata.get("title", metadata.get("headline")),
                    }
                })
            # Create new result
            new_result = {
                "id": idx,
                "user_input": query,
                "reference_source_id": item['source_id'],
                "reference_context": item['context'],
                "answer_reference": item['answer'],
                "retrieved_contexts": retrieved_contexts
            }
            
            # Append to results and save immediately
            results.append(new_result)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"Error processing item {idx}: {str(e)}")
            continue

except KeyboardInterrupt:
    print("\nProcessing interrupted by user. Partial results have been saved.")
except Exception as e:
    print(f"\nUnexpected error: {str(e)}")
finally:
    print(f"\nResults saved to {output_file}")