from traditional_metrics import compute_precision, compute_recall, compute_mrr
import json
import argparse

def evaluate_retrieval(eval_dataset, method='paragraph', logging=False):
    results = {
        'precision': [],
        'recall': [], 
        'mrr': []
    }
    
    print(f"\nProcessing {len(eval_dataset)} evaluation items...")
    
    for eval_item in eval_dataset:
        if method == 'paragraph':
            reference_sources = [f"{eval_item['reference_source_id']}-{ref_paragraph}" for ref_paragraph in eval_item['reference_context_paragraphs']]
            retrieved_sources = [f"{ctx['context_metadata']['source_id']}-{ctx['context_metadata']['paragraph_position']}" for ctx in eval_item['retrieved_contexts']]
            deduplicate = False
        elif method == 'document':
            reference_sources = [eval_item['reference_source_id']]
            retrieved_sources = [f"{ctx['context_metadata']['source_id']}" for ctx in eval_item['retrieved_contexts']]
            deduplicate = True 

        precision = compute_precision(reference_sources, retrieved_sources, deduplicate=deduplicate)
        recall = compute_recall(reference_sources, retrieved_sources, deduplicate = deduplicate)
        mrr = compute_mrr(reference_sources, retrieved_sources)
        if logging:
            print(f"Evaluating item {eval_item['id']} with method '{method}'")
            print(f"  Reference Sources: {reference_sources}")
            print(f"  Retrieved Sources: {retrieved_sources}")
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
parser.add_argument('--logging', action='store_true', help='Enable debug mode')
args = parser.parse_args()
# Load the data
print("Loading datasets...")
with open(args.results) as f:
    eval_dataset = json.load(f)
    
# Run evaluation
results_paragraph = evaluate_retrieval(eval_dataset, method='paragraph', logging=args.logging)
results_document = evaluate_retrieval(eval_dataset, method='document', logging=args.logging)
print("\n=== Final Results by Paragraph ===")
print(f"Average Precision: {results_paragraph['avg_precision']:.3f}")
print(f"Average Recall: {results_paragraph['avg_recall']:.3f}")
print(f"Average MRR: {results_paragraph['avg_mrr']:.3f}")  
print("=== Final Results by Document ===")
print(f"Average Precision: {results_document['avg_precision']:.3f}")
print(f"Average Recall: {results_document['avg_recall']:.3f}")
print(f"Average MRR: {results_document['avg_mrr']:.3f}") 

