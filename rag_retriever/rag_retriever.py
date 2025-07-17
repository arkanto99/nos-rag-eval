import os
import sys
from pathlib import Path

# Add the rag directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from elasticsearch import Elasticsearch
import torch
from configs.ConfigLoader import ConfigLoader
from retriever.Reranker import Reranker
from retriever.Retriever import Retriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_elasticsearch import ElasticsearchStore, ElasticsearchRetriever
from typing import List, Dict, Any, Tuple
from enum import Enum
import pprint


class ElasticSearchStrategy(Enum):
    BM25 = ElasticsearchStore.BM25RetrievalStrategy()
    APROX = ElasticsearchStore.ApproxRetrievalStrategy()
    EXACT = ElasticsearchStore.ExactRetrievalStrategy()
    SPARSE = ElasticsearchStore.SparseVectorRetrievalStrategy()

class RAG:
    def __init__(self, config_file):
        """
        Initialize the Retriever and Reranker mode.
        """
        # Initialize retriever
        print("Initializing retriever...")
        self.config = ConfigLoader.load(config_file)
        self.elastic_config = ConfigLoader.load_elastic(self.config.database.elastic_config_file)
        self.retriever = self.__initialize_retriever()
        print("RAG system initialized successfully.")
    
    def __initialize_retriever(self):
        strategy_name = self.config.retriever.retrieval_strategy
        retrieval_strategy = ElasticSearchStrategy[strategy_name].value

        # vectorstore = ElasticsearchStore(
        #         es_url=elastic_config.endpoint,
        #         es_user=elastic_config.username,
        #         es_password=elastic_config.password,
        #         index_name=elastic_config.elastic_index,
        #         strategy=retrieval_strategy,
        # )

        #https://python.langchain.com/docs/integrations/retrievers/elasticsearch_retriever/
        def bm25_query(search_query: str) -> Dict:
            return {
                "query": {
                    "match": {
                        "text": search_query,
                    },
                },
            }
        es_client = Elasticsearch(
            hosts=[self.elastic_config.endpoint],
            basic_auth=(self.elastic_config.username, self.elastic_config.password)
        )
        vectorstore_retriever = ElasticsearchRetriever(
            es_client=es_client,
            index_name=self.elastic_config.elastic_index,
            content_field="text",
            body_func=bm25_query,
        )
        reranker = Reranker(
            model_name=self.config.reranker.reranker_model,
            hf_cache_dir=self.config.general_config.hf_cache_dir,
            use_fp16=True,  # Use half-precision for efficiency
            normalize=True  # Normalize scores to 0-1 range
        ) if self.config.reranker.use_reranking else None

        return Retriever(
            vectorstore=vectorstore_retriever,
            top_k=self.config.retriever.query_top_k,
            reranker=reranker,
            initial_retrieve_count=self.config.retriever.initial_retrieve_count
        )

    def retrieve_contexts(self, user_query: str):        
        # Retrieve relevant documents
        initial_docs, final_docs = self.retriever.invoke(user_query)
        # Format context from retrieved and reranked documents in Galician
        context = "\n\n".join([f"Documento {i+1}: {doc.page_content}" for i, (doc,_) in enumerate(final_docs)])
        
        # Store source information
        source_info = []
        initial_docs_info = []
        for i, (doc,score) in enumerate(final_docs):
            # Get document content and metadata
            source_data = {
                "id": i+1,
                "score": score,
                "content": doc.page_content,
                "metadata": doc.metadata if hasattr(doc, "metadata") else {}
            }
            source_info.append(source_data)
        for i, (doc,score) in enumerate(initial_docs):
            # Get document content and metadata
            source_data = {
                "id": i+1,
                "score": score,
                "content": doc.page_content,
                "metadata": doc.metadata if hasattr(doc, "metadata") else {}
            }
            initial_docs_info.append(source_data)

        return source_info, initial_docs_info

if __name__ == "__main__":
    # Print the configuration for debugging
    pp = pprint.PrettyPrinter(indent=4)
    rag = RAG("configs/config.yaml")
    pp.pprint(rag.config.__dict__)
    print("Asistente RAG en Galego (escriba 'sair' para rematar)")
    while True:
        user_input = input("\nUsuario: ")
        if user_input.lower() in ["sair", "quit", "exit"]:
            print("Grazas por usar o asistente!")
            break
            
        # Get response from RAG system
        sources, initial_docs_info= rag.retrieve_contexts(
            user_input
        )
        print_sources = True
        if print_sources:
            print("\n--- Fragmentos empregados ---")
            for source in sources:
                #print(source)
                source_id = source["id"]
                content = source["content"]
                metadata = source["metadata"]["_source"]
                # If you have source info in metadata, you can display it
                source_file = metadata.get("source_id",f"Praza-{metadata.get('published_on')}")
                
                print(f"\nFragmento {source_id} - {source_file}")
                print("-" * 40)
                print(content)
                print("-" * 40)