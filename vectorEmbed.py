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
        
        # Modified ID format to support prefix queries better
        vectors = [{
            'id': f"{source_id}#{i}",  # Changed separator from _ to # for better prefix matching
            'values': embedding,
            'metadata': {'text': chunk, 'source': source_id}
        } for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))]
        
        self.index.upsert(vectors=vectors)

    def query_database(self, query: str, top_k: int = 100):
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

    def query_database_by_prefix(self, prefix: str, top_k: int = 100):
        try:
            # First, get all IDs that match the prefix
            matching_ids = list(self.index.list(prefix=prefix))
            
            if not matching_ids:
                return []
            
            # Fetch the actual vectors using these IDs
            fetch_response = self.index.fetch(ids=matching_ids)
            
            # Convert fetch response to list of results
            results = []
            for id, vector in fetch_response.vectors.items():
                results.append({
                    'text': vector.metadata['text'],
                    'source': vector.metadata.get('source', 'Unknown'),
                    'id': id
                })
            
            # Sort results by relevance if needed
            # Currently returning all matches up to top_k
            if top_k is not None:
                results = results[:top_k]
                
            return results
            
        except Exception as e:
            print(f"Error in prefix query: {str(e)}")
            return []
