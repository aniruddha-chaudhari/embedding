import os
from typing import List, Union
from pinecone import Pinecone, ServerlessSpec
import cohere

class VectorDatabase:
    def __init__(self):
        self.co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = "coffeeshop-index"
        self.model_name = 'embed-english-light-v3.0'
        self.dimension = 384
        
        try:
            try:
                existing_index = self.pc.describe_index(self.index_name)
                if (existing_index.dimension != self.dimension):
                    self.pc.delete_index(self.index_name)
            except:
                pass

            self.index = self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        except Exception:
            self.index = self.pc.Index(self.index_name)

    def get_embedding(self, text: Union[str, List[str]]):
        texts = [text] if isinstance(text, str) else text
        response = self.co.embed(
            texts=texts,
            model=self.model_name,
            input_type='search_document',
            embedding_types=['float']
        )
        return [list(e) for e in response.embeddings.float_]

    def add_content_to_database(self, content: str, source_id: str, chunk_size: int = 1000):
        chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
        embeddings = self.get_embedding(chunks)
        
        if not embeddings or not all(isinstance(e, list) and all(isinstance(v, float) for v in e) for e in embeddings):
            raise ValueError("Invalid embedding format")
        
        vectors = [{
            'id': f"{source_id}_{i}",
            'values': embedding,
            'metadata': {'text': chunk, 'source': source_id}
        } for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))]
        
        try:
            # Try to delete existing namespace
            self.index.delete(delete_all=True, namespace=source_id)
            print(f"Deleted existing namespace: {source_id}")
        except Exception as e:
            print(f"Namespace deletion failed or doesn't exist: {str(e)}")
        
        # Add new vectors regardless of whether deletion succeeded
        self.index.upsert(
            vectors=vectors,
            namespace=source_id
        )
        print(f"Added {len(vectors)} vectors to namespace: {source_id}")

    def query_database(self, query: str, source):
        query_embedding = self.get_embedding(query)[0]
        
        results = self.index.query(
            namespace=source,
            vector=query_embedding,
            top_k=10000,
            include_values=False,
            include_metadata=True,
        )
        
        return [{
            'text': match.metadata['text'],
            'source': match.metadata.get('source', 'Unknown'),
            'score': match.score
        } for match in results.matches]
