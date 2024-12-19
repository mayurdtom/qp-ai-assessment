from pymilvus import MilvusClient, Collection, CollectionSchema, FieldSchema, DataType, utility, connections
from config.milvus_config import MILVUS_CONFIG

class MilvusVectorStore:
    def __init__(self):
        self._connect_to_milvus()
        self.ensure_collection_exists()

    def _connect_to_milvus(self):
        """Connect to the Milvus server."""
        connections.connect(
            alias="default", 
            host=MILVUS_CONFIG['host'],
            port=MILVUS_CONFIG['port']
        )
        print(f"Connected to Milvus server at {MILVUS_CONFIG['host']}:{MILVUS_CONFIG['port']}")

    def ensure_collection_exists(self):
        collection_name = MILVUS_CONFIG['collection_name']

        # Check if the collection exists
        if not utility.has_collection(collection_name):
            print(f"Collection '{collection_name}' not found. Creating it...")
            
            # Define schema
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=1000),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=MILVUS_CONFIG['dimension']),
            ]
            schema = CollectionSchema(fields, description="Stores document chunks and embeddings")
            
            # Create the collection
            collection = Collection(name=collection_name, schema=schema)
            print(f"Collection '{collection_name}' created.")
            
            # Create an index on the embedding field
            self.create_index(collection)
        else:
            print(f"Collection '{collection_name}' already exists.")
    
    def create_index(self, collection):
        """Create an index on the embedding field."""
        index_params = {
            "index_type": "IVF_FLAT",  # Index type (use IVF_FLAT for simplicity, others like IVF_SQ8, HNSW are also possible)
            "metric_type": "L2",        # Use L2 distance for similarity search
            "params": {"nlist": 100}    # Number of clusters, you can adjust based on your data
        }
        
        # Create the index
        collection.create_index(field_name="embedding", index_params=index_params)
        print("Index created successfully on 'embedding' field.")
    
    def insert_embeddings(self, chunks, embeddings):
        collection_name = MILVUS_CONFIG['collection_name']
        collection = Collection(collection_name)

        data = [
            chunks, 
            embeddings
        ]

        collection.insert(data)
        print("Embeddings inserted successfully.")
    
    def semantic_search(self, query_embedding, top_k=3):
        collection_name = MILVUS_CONFIG['collection_name']
        collection = Collection(collection_name)
        collection.load()

        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        
        search_results = collection.search(
            data=[query_embedding], 
            anns_field="embedding", 
            param=search_params, 
            limit=top_k,
            output_fields=["chunk_text"],
            expr=None,
            consistency_level="Strong"
        )

        # Process and structure the results
        results = []
        for hits in search_results:
            for hit in hits:
                result = {
                    'id': hit.id,
                    'distance': hit.distance,
                    'text': hit.entity.value_of_field('chunk_text')
                }
                results.append(result)

        return results
