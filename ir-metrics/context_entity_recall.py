from span_marker import SpanMarkerModel
import json
from typing import List, Dict, Set
import random
import torch

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

def analyze_example(model, example: Dict) -> Dict:
    """Analyze a single example and return metrics."""
    # Get reference entities
    reference_entities = extract_entities(model, example['reference'])

    # Get retrieved context entities
    retrieved_entities = set()
    for context in example['retrieved_contexts']:
        context_entities = extract_entities(model, context)
        retrieved_entities.update(context_entities)
    
    # Compute recall
    recall = compute_entity_recall(reference_entities, retrieved_entities)
    
    return {
        'question': example['user_input'],
        'reference_entities': list(reference_entities),
        'retrieved_entities': list(retrieved_entities),
        'entity_recall': recall
    }

if __name__ == "__main__":
    # Load model
    model = get_entity_extractor_model()
    
    # Load dataset
    with open('/home/compartido/pabloF/nos-rag-eval/datasets/evaluation_dataset_fixed.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # Select 5 random examples
    sample_size = 5
    examples = random.sample(dataset, sample_size)
    
    # Process examples
    results = []
    print(f"\nAnalyzing {sample_size} random examples...")
    
    for i, example in enumerate(examples, 1):
        result = analyze_example(model, example)
        results.append(result)
        
        # Print results for this example
        print(f"\nExample {i}:")
        print(f"Question: {result['question']}")
        print(f"Reference entities: {result['reference_entities']}")
        print(f"Retrieved entities: {result['retrieved_entities']}")
        print(f"Entity recall: {result['entity_recall']:.4f}")
    
    # Calculate average recall for these examples
    avg_recall = sum(r['entity_recall'] for r in results) / len(results)
    print(f"\nAverage entity recall for {sample_size} examples: {avg_recall:.4f}")