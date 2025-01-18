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
        
        self.index.upsert(vectors=vectors)

    def query_database(self, query: str, top_k: int = 3):
        query_embedding = self.get_embedding(query)[0]
        # Use 10000 as a practical upper limit when top_k is None
        fetch_limit = 10000 if top_k is None else top_k
        results = self.index.query(
            vector=query_embedding,
            top_k=fetch_limit,
            include_values=False,
            include_metadata=True
        )
        
        return [{
            'text': match.metadata['text'],
            'source': match.metadata.get('source', 'Unknown'),
            'score': match.score
        } for match in results.matches]

    def query_database_by_prefix(self, prefix: str, top_k: int = 3):
        # Use 10000 as a practical upper limit when top_k is None
        fetch_limit = 10000 if top_k is None else top_k
        response = self.index.query(
            vector=[0] * self.dimension,
            top_k=fetch_limit,
            include_values=False,
            include_metadata=True
        )
        
        prefix_matches = [
            match for match in response.matches 
            if match.metadata['text'].lower().startswith(prefix.lower())
        ]
        
        if top_k is not None:
            prefix_matches = prefix_matches[:top_k]
        
        return [{
            'text': match.metadata['text'],
            'source': match.metadata.get('source', 'Unknown'),
            'score': match.score
        } for match in prefix_matches]
