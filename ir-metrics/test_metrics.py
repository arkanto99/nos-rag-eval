from traditional_metrics import compute_precision, compute_recall, compute_mrr
import json
import argparse

def evaluate_retrieval(eval_dataset):
    results = {
        'precision': [],
        'recall': [], 
        'mrr': []
    }
    
    print(f"\nProcessing {len(eval_dataset)} evaluation items...")
    
    for eval_item in eval_dataset:

        reference_source = eval_item["reference_source_id"]
        retrieved_sources = [ctx['context_metadata']["source_id"] for ctx in eval_item['retrieved_contexts']]

        print(f"\n--- Processing item {eval_item['id']} ---")
 
        precision = compute_precision([reference_source], retrieved_sources)
        recall = compute_recall([reference_source], retrieved_sources)
        mrr = compute_mrr([reference_source], retrieved_sources)
            
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  MRR: {mrr:.3f}")
            
        results['precision'].append(precision)
        results['recall'].append(recall)
        results['mrr'].append(mrr)
    
    # Calculate averages
    avg_results = {
        'avg_precision': sum(results['precision']) / len(results['precision']) if results['precision'] else 0,
        'avg_recall': sum(results['recall']) / len(results['recall']) if results['recall'] else 0,
        'avg_mrr': sum(results['mrr']) / len(results['mrr']) if results['mrr'] else 0
    }
    return avg_results

parser = argparse.ArgumentParser(description="Evaluate with traditional metrics Retrieval Results")
parser.add_argument('--results', type=str, default=None, help='Path to results')
args = parser.parse_args()
# Load the data
print("Loading datasets...")
with open(args.results) as f:
    eval_dataset = json.load(f)
    
# Run evaluation
results = evaluate_retrieval(eval_dataset)
print("\n=== Final Results ===")
print(f"Average Precision: {results['avg_precision']:.3f}")
print(f"Average Recall: {results['avg_recall']:.3f}")
print(f"Average MRR: {results['avg_mrr']:.3f}")