from FlagEmbedding import FlagReranker
from typing import List, Any

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
        self.reranker = FlagReranker(model_name, cache_dir=hf_cache_dir, use_fp16=use_fp16)
        
    def compute_scores(self, query: str, passages: List[str]) -> List[float]:
        # Create pairs of [query, passage] for each passage
        pairs = [[query, passage] for passage in passages]
        
        scores = self.reranker.compute_score(pairs, normalize=self.normalize)
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