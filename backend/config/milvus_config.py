from pymilvus import CollectionSchema

MILVUS_CONFIG = {
    'host': 'localhost',
    'port': '19530',
    'collection_name': 'document_chunks',
    'dimension': 384,  
    'index_type': 'IVF_FLAT',
    'metric_type': 'COSINE'
}

