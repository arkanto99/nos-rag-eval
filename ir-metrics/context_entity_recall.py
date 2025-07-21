from span_marker import SpanMarkerModel
import json
import os
from typing import List, Dict, Set
import torch
from tqdm import tqdm
import glob

def patch():
    from span_marker.configuration import SpanMarkerConfig
    
    def patch_getattribute(self, key: str):
        try:
            return super(SpanMarkerConfig, self).__getattribute__(key)
        except AttributeError as e:
            try:
                return super(SpanMarkerConfig, self).__getattribute__("encoder")[key]
            except KeyError:
                raise e
            except TypeError:
                raise e
    
    SpanMarkerConfig.__getattribute__ = patch_getattribute

def get_entity_extractor_model():
    patch()
    """Load the SpanMarkerModel from the pretrained model."""
    model = SpanMarkerModel.from_pretrained("tomaarsen/span-marker-mbert-base-multinerd",
                                          cache_dir="/home/compartido/pabloF/cache",
                                          trust_remote_code=True,
                                          torch_dtype=torch.float32)
    model.cuda()
    return model

def get_entity_set(entities_list: List[Dict]) -> Set[str]:
    """Convert list of entity dictionaries to a set of entity texts."""
    return set(entity['span'] for entity in entities_list)
    
def extract_span(model, text: str) -> List[Dict]:
    """Extract entities from text using the model."""
    entities = model.predict(text)
    return entities

def extract_entities(model, text: str) -> Set[str]:
    """Extract entities from text and return as a set of entity texts."""
    entities = extract_span(model, text)
    if not entities:
        return set()
    return get_entity_set(entities)

def compute_entity_recall(reference_entities: Set[str], retrieved_entities: Set[str]) -> float:
    """Compute Context Entity Recall metric."""
    if not reference_entities:
        return 0.0
    common_entities = reference_entities.intersection(retrieved_entities)
    recall = len(common_entities) / len(reference_entities)
    return recall

def analyze_all_examples(model, example: Dict) -> Dict:
    """Analyze a single example and return metrics."""
    # Get reference entities
    reference_entities = extract_entities(model, example['reference_context'])

    # Get retrieved context entities
    retrieved_entities = set()
    for ir_context in example['retrieved_contexts']:
        context_entities = extract_entities(model, ir_context['context'])
        retrieved_entities.update(context_entities)
    
    # Compute recall
    recall = compute_entity_recall(reference_entities, retrieved_entities)
    
    return {
        'question': example['user_input'],
        'reference_entities': list(reference_entities),
        'retrieved_entities': list(retrieved_entities),
        'entity_recall': recall
    }

def analyze_entities(model, example: Dict) -> Dict:
    """Analyze a single example and return metrics."""
    # Get reference entities
    reference_entities = extract_entities(model, example['reference_context'])

    # Get retrieved context entities
    retrieved_entities = set()
    first_context_entities = extract_entities(model, example['retrieved_contexts'][0]['context'])
    retrieved_entities.update(first_context_entities)
    for ir_context in example['retrieved_contexts'][1:]:
        context_entities = extract_entities(model, ir_context['context'])
        retrieved_entities.update(context_entities)
    
    # Compute recall
    first_recall = compute_entity_recall(reference_entities, first_context_entities)
    recall = compute_entity_recall(reference_entities, retrieved_entities)
    
    return {
        'question': example['user_input'],
        'reference_entities': list(reference_entities),
        'retrieved_entities': list(retrieved_entities),
        'entity_recall_first_context': first_recall,
        'total_entity_recall': recall
    }

if __name__ == "__main__":
    # Load model
    model = get_entity_extractor_model()
    
    # Load dataset
    # with open('/home/compartido/pabloF/nos-rag-eval/results/retrieved_dataset_all-minilm-l6-v2_chunked.json', 'r', encoding='utf-8') as f:
    #     dataset = json.load(f)
    results_dir = "/home/compartido/pabloF/nos-rag-eval/results"
    json_files = glob.glob(f"{results_dir}/retrieved_dataset_*.json")
    out_file = f"all_entity_recall_results.json"
    # Initialize or load previous results if file exists
    if os.path.exists(out_file):
        with open(out_file, 'r', encoding='utf-8') as out_f:
            all_results = json.load(out_f)
    else:
        all_results = []

    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        # Process examples
        results = []
        print(f"\nAnalyzing {len(dataset)} examples of {os.path.basename(json_file)}...")
        
        for example in tqdm(dataset):
            result = analyze_entities(model, example)
            results.append(result)
            
            # Print results for this example
            # print(f"\nExample {i}:")
            # print(f"Question: {result['question']}")
            # print(f"Reference entities: {result['reference_entities']}")
            # print(f"Retrieved entities: {result['retrieved_entities']}")
            # print(f"Entity recall: {result['entity_recall']:.4f}")
        
        # Calculate average recall for these examples
        avg_recall_first_context = sum(r['entity_recall_first_context'] for r in results) / len(results)
        avg_recall = sum(r['total_entity_recall'] for r in results) / len(results)
        print("\n--- Entity Recall Results ---")
        print(f"Average entity recall in the first context: {avg_recall_first_context :.4f}")
        print(f"Average entity recall: {avg_recall:.4f}")
        all_results.append({
            "file": os.path.basename(json_file),
            "average_entity_recall_first_context": avg_recall_first_context,
            "average_entity_recall": avg_recall
        })

        # Save results to file after each file is processed
        with open(out_file, 'w', encoding='utf-8') as out_f:
            json.dump(all_results, out_f, ensure_ascii=False, indent=2)
        print(f"Appended recall results to {out_file}")