# Code from https://discuss.huggingface.co/t/use-ragas-with-huggingface-llm/75769

import sys
import os
import json
from tqdm import tqdm

# Get the current directory and set up paths
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # nos-rag-eval
parent_dir = os.path.dirname(current_dir)  # pabloF
rag_dir = os.path.join(parent_dir, 'rag')  # rag directory

# Add only rag path since that's where our module is
sys.path.append(rag_dir)

# Import directly from rag_ragas
from rag_ragas import RAG
from dataloader_evaluation import load_qa_with_metadata

dataset = []
dataset = load_qa_with_metadata(file_path="/home/compartido/pabloF/nos-rag-eval/datasets/qwen_samples_context_fixed.json")
rag = RAG()
#sample_queries = sample_queries[:10]  # Limit to first 10 queries for testing
#expected_responses = expected_responses[:10]  # Limit to first 10 responses for testing

# ...existing imports...

# Initialize or load existing results
output_file = 'evaluation_dataset_with_metadata.json'
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
        reference = item['answer']
        
        try:
            _, relevant_docs = rag.retriever.invoke(query)
            response = rag.generate_answer(query, relevant_docs)
            
            # Create new result
            new_result = {
                "id": idx,
                "user_input": query,
                "retrieved_contexts": [doc.page_content for doc, _ in relevant_docs],
                "response": response,
                "reference": reference
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