from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import torch
import json
import os
from pathlib import Path
from tqdm import tqdm
import argparse

class ElasticSearchProxy():
    def __init__(self, index_path, embedding_model, hf_cache_dir):
        self.es = Elasticsearch(
            hosts=["http://localhost:9200"],
            basic_auth=("elastic", "eZr4ltvT"),
        )    
        with open(index_path, 'r', encoding='utf-8') as f:
            index_config = json.load(f)
        self.index = index_config["index_name"]
        self.mapping = index_config["mapping"]
        if embedding_model:
            self.embedding_model = SentenceTransformer(embedding_model, cache_folder=hf_cache_dir)
            self.embedding_model.to('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.embedding_model = None

    def create_index(self):
        if not self.es.indices.exists(index=self.index):
            self.es.indices.create(index=self.index, body=self.mapping)
            print(f"Created index {self.index} with mapping")

    def index_article(self, article):
        if self.embedding_model:
            article['text_embedding'] = self.embedding_model.encode(article['text'])
        self.es.index(index=self.index, document=article)

    def index_json_files(self, data_path):
        for json_file in data_path.glob("*.jsonl"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    # Count lines for tqdm
                    total_lines = sum(1 for line in f if line.strip())
                    f.seek(0)  # Reset file pointer to beginning
                    # Create progress bar
                    pbar = tqdm(total=total_lines, desc=f"Indexing {json_file.name}")
                    for line in f:
                        if line.strip():  # Skip empty lines
                            article = json.loads(line)
                            self.index_article(article)
                            pbar.update(1)
                    pbar.close()     
                print(f"\nSuccessfully processed {json_file.name}")
            except json.JSONDecodeError as e:
                print(f"Error: {json_file.name} contains invalid JSON: {e}")
            except Exception as e:
                print(f"Error processing {json_file.name}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Elasticsearch index with embeddings.")
    parser.add_argument('--index', type=str, default=None, help='Path to the Elasticsearch index')
    parser.add_argument('--embedding', type=str, default=None, help='Path to the embedding model')
    parser.add_argument('--hf_cache_dir', type=str, default=None, help='Path to Hugging Face cache directory')
    parser.add_argument('--data_path', type=str, default="data/combined_datasets",help="Path to the data directory containing JSON files")
    args = parser.parse_args()
    print(args)
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Error: Data directory not found at {data_path}")
    else:
        esp = ElasticSearchProxy(args.index, args.embedding, args.hf_cache_dir)
        print(f"Creating index {args.index} with mapping...")
        esp.create_index()
        print(f"Processing JSON files from {data_path}")
        esp.index_json_files(data_path)
        print("\nIndexing complete!")