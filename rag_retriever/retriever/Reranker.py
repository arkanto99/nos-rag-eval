from FlagEmbedding import FlagReranker
from typing import List, Any
from sentence_transformers import SentenceTransformer
import numpy as np

class SentenceTransformerReranker(SentenceTransformer):
    def __init__(self, model_name: str, cache_dir: str = None, use_fp16: bool = True):
        """
        Args:
            model_name: Name of the SentenceTransformer model to use
            use_fp16: Whether to use half-precision for inference
            normalize: Whether to normalize scores to 0-1 range using sigmoid
        """
        super().__init__(model_name, cache_folder=cache_dir, device='cuda' if use_fp16 else 'cpu')

    def compute_scores(self, query: str, passages: List[str], normalize) -> List[float]:
        # Encode query and passages
        query_embedding = self.encode(query, convert_to_tensor=True)
        passage_embeddings = self.encode(passages, convert_to_tensor=True)
        
        # Compute cosine similarities
        scores = (query_embedding @ passage_embeddings.T).cpu().numpy().tolist()
        
        if normalize:
            scores = [1 / (1 + np.exp(-score)) for score in scores]  # Sigmoid normalization
        return scores

class FlagEmbeddingReranker(FlagReranker):
    def __init__(self, model_name: str, cache_dir: str = None, use_fp16: bool = True):
        """
        Args:
            model_name: Name of the FlagEmbedding model to use
            cache_dir: Directory to cache the model files
            use_fp16: Whether to use half-precision for inference
        """
        super().__init__(model_name, cache_dir=cache_dir, use_fp16=use_fp16)

    def compute_scores(self, query: str, passages: List[str], normalize) -> List[float]:
        # Create pairs of [query, passage] for each passage
        pairs = [[query, passage] for passage in passages]
        scores = self.compute_score(pairs, normalize=normalize) #Using FlagEmbedding's compute_score method
        return scores


class Reranker:
    def __init__(self, model_name, hf_cache_dir, use_fp16=True, normalize=True):
        """
        Args:
            model_name: Name of the reranker model to use
            use_fp16: Whether to use half-precision for inference
            normalize: Whether to normalize scores to 0-1 range using sigmoid
        """
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.normalize = normalize
        if model_name in ["BAAI/bge-reranker-v2-m3"]:
            # Use FlagEmbedding for BGE reranker
            self.reranker = FlagEmbeddingReranker(model_name, cache_dir=hf_cache_dir, use_fp16=self.use_fp16)
        else:
            # Use SentenceTransformer for other models
            self.reranker = SentenceTransformerReranker(model_name, cache_dir=hf_cache_dir, use_fp16=self.use_fp16)
        
    def compute_scores(self, query: str, passages: List[str]) -> List[float]:
        # Create pairs of [query, passage] for each passage
        scores = self.reranker.compute_scores(query, passages, normalize=self.normalize)
        return scores
    
    def rerank(self, query: str, docs: List[Any], top_k: int = 5) -> List[Any]:
        """
        Rerank documents based on relevance to query.
        
        Args:
            query: User query
            docs: List of documents to rerank
            top_k: Number of documents to return
            
        Returns:
            List of reranked documents
        """
        # Extract text content from documents
        passages = [doc.page_content for doc in docs]
        scores = self.compute_scores(query, passages)
        
        # Create (doc, score) pairs and sort by score in descending order
        scored_docs = list(zip(docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        #return [doc for doc, _ in scored_docs[:top_k]]
        return scored_docs