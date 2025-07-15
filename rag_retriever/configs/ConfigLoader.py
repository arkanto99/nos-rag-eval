from dataclasses import dataclass
import yaml
import os

@dataclass
class GeneralConfig:
    hf_cache_dir: str

@dataclass
class DatabaseConfig:
    chunk_size: int
    chunk_overlap: int
    elastic_config_file: str

@dataclass
class RetrieverConfig:
    embedding_model: str
    retrieval_strategy: str
    initial_retrieve_count: int
    query_top_k: int

@dataclass
class RerankerConfig:
    use_reranking: bool
    reranker_model: str

@dataclass
class ElasticConfig:
    username: str
    password: str
    elastic_index: str
    url: str
    endpoint: str

@dataclass
class Config:
    general_config: GeneralConfig
    database: DatabaseConfig
    retriever: RetrieverConfig
    reranker: RerankerConfig
    

class ConfigLoader:
    @staticmethod
    def load() -> Config:
        # Automatically determine the path to config.yaml in the same directory as this script
        script_dir = os.path.dirname(__file__)
        config_path = os.path.join(script_dir, "config.yaml")
        
        with open(config_path, 'r') as file:
            config_dict = yaml.safe_load(file)
        
        return Config(
            general_config=GeneralConfig(**config_dict['general_config']),
            database=DatabaseConfig(**config_dict['database']),
            retriever=RetrieverConfig(**config_dict['retriever']),
            reranker=RerankerConfig(**config_dict['reranker'])
        )
    @staticmethod
    def load_elastic(config_path = "config_elastic.yaml") -> Config:
        script_dir = os.path.dirname(__file__)
        config_path = os.path.join(script_dir, config_path)
        with open(config_path, 'r') as file:
            config_dict = yaml.safe_load(file)
        return ElasticConfig(
            username=config_dict['username'],
            password=config_dict['password'],
            elastic_index=config_dict['elastic_index'],
            url=config_dict['elastic_url'],
            endpoint=config_dict['api_endpoint']
        )

if __name__ == "__main__":
    config_path = "config.yaml"
    config = ConfigLoader.load(config_path)
    print("General Config:", config.general_config)
    print("Database Config:", config.database)
    print("Retriever Config:", config.retriever)
    print("Reranker Config:", config.reranker)