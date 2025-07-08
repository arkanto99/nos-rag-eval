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
from dataloader_evaluation import load_qa_pairs


rag = RAG()
dataset = []

sample_queries,expected_responses = load_qa_pairs(file_path="/home/compartido/pabloF/nos-rag-eval/datasets/qwen_samples_context_fixed.json")

#sample_queries = sample_queries[:10]  # Limit to first 10 queries for testing
#expected_responses = expected_responses[:10]  # Limit to first 10 responses for testing

for query,reference in tqdm(zip(sample_queries, expected_responses), 
                           total=len(sample_queries),
                           desc="Generating dataset"):

    _, relevant_docs = rag.retriever.invoke(query)
    response = rag.generate_answer(query, relevant_docs)
    print(relevant_docs)
    dataset.append(
        {
            "user_input":query,
            "retrieved_contexts":[doc.page_content for doc, _ in relevant_docs],
            "response":response,
            "reference":reference
        }
    )

# Save dataset to file
with open('evaluation_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)