import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import os
import torch
from typing import List, Dict, Any

class HybridRetriever:
    def __init__(self, 
                 dense_model_name='all-MiniLM-L6-v2', 
                 rerank_model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        """
        Initializes the HybridRetriever with specified models.
        """
        # Detect device
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        print(f"Loading dense model: {dense_model_name}...")
        self.dense_model = SentenceTransformer(dense_model_name, device=self.device)
        
        print(f"Loading reranker model: {rerank_model_name}...")
        self.reranker = CrossEncoder(rerank_model_name, device=self.device)
        
        self.documents = []
        self.doc_ids = []
        self.index = None
        self.bm25 = None
        
    def load_data(self, data_path: str):
        """
        Loads documents from the specified JSON file.
        The expected format is:
        {
            "topic_id": ...,
            "docs": [
                {"id": "...", "content": "...", ...},
                ...
            ]
        }
        However, based on the README, the docs.json file contains multiple records, possibly one per line or a list of objects.
        Let's verify the format of docs.json first.
        """
        print(f"Loading data from {data_path}...")
        self.documents = []
        self.doc_ids = []
        
        # Check if file exists
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"File not found: {data_path}")
            
        # Assuming docs.json is a list of topic objects based on typical JSON structure
        # Or it could be line-delimited JSON. Let's try standard JSON load first.
        try:
            with open(data_path, 'r') as f:
                data = json.load(f)
                
            # Flatten documents from all topics
            # If data is a list of topics
            if isinstance(data, list):
                for topic in data:
                    if 'docs' in topic:
                        for doc in topic['docs']:
                            self.documents.append(doc)
                            self.doc_ids.append(doc['id'])
            # If data is a dict (maybe single topic?)
            elif isinstance(data, dict):
                 if 'docs' in data:
                        for doc in data['docs']:
                            self.documents.append(doc)
                            self.doc_ids.append(doc['id'])
            else:
                print("Unknown data format.")
                
        except json.JSONDecodeError:
            # Maybe it's jsonl?
            print("Standard JSON load failed, trying JSONL...")
            with open(data_path, 'r') as f:
                for line in f:
                    try:
                        topic = json.loads(line)
                        if 'docs' in topic:
                            for doc in topic['docs']:
                                self.documents.append(doc)
                                self.doc_ids.append(doc['id'])
                    except:
                        continue
        
        print(f"Loaded {len(self.documents)} documents.")

    def build_indices(self):
        """
        Builds both Dense (FAISS) and Sparse (BM25) indices.
        """
        if not self.documents:
            print("No documents to index.")
            return

        # 1. Sparse Index (BM25)
        print("Building Sparse Index (BM25)...")
        tokenized_corpus = [doc['content'].split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # 2. Dense Index (FAISS)
        print("Building Dense Index (FAISS)...")
        corpus_embeddings = self.dense_model.encode([doc['content'] for doc in self.documents], show_progress_bar=True)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(corpus_embeddings)
        
        d = corpus_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d) # Inner Product (sim to Cosine after normalization)
        self.index.add(corpus_embeddings)
        
        print("Indexing complete.")

    def retrieve(self, query: str, top_k_dense=20, top_k_sparse=20, top_k_final=5) -> List[Dict[str, Any]]:
        """
        Performs hybrid retrieval and reranking.
        """
        if not self.documents:
            print("No documents loaded.")
            return []

        # 1. Dense Retrieval
        query_embedding = self.dense_model.encode([query])
        faiss.normalize_L2(query_embedding)
        D, I = self.index.search(query_embedding, top_k_dense)
        dense_results = set(I[0])

        # 2. Sparse Retrieval
        tokenized_query = query.split()
        sparse_scores = self.bm25.get_scores(tokenized_query)
        sparse_indices = np.argsort(sparse_scores)[::-1][:top_k_sparse]
        sparse_results = set(sparse_indices)

        # 3. Combine Candidates
        candidate_indices = list(dense_results.union(sparse_results))
        
        # 4. Reranking
        candidates = [self.documents[i] for i in candidate_indices]
        sentence_pairs = [[query, doc['content']] for doc in candidates]
        
        rerank_scores = self.reranker.predict(sentence_pairs)
        
        # Associate scores with candidates
        results_with_scores = []
        for i, score in enumerate(rerank_scores):
            results_with_scores.append({
                "doc": candidates[i],
                "score": float(score)  # Convert float32 to python float for json serialization
            })
            
        # Sort by rerank score descending
        results_with_scores.sort(key=lambda x: x['score'], reverse=True)
        
        return results_with_scores[:top_k_final]

if __name__ == "__main__":
    # Example Usage
    dataset_path = "semeval2026-task12-dataset/dev_data/docs.json"
    
    retriever = HybridRetriever()
    retriever.load_data(dataset_path)
    retriever.build_indices()
    
    # Sample Query
    query = "President Yoon Suk Yeol vowed to carry out a thorough investigation."
    print(f"\nQuery: {query}")
    results = retriever.retrieve(query)
    
    for rank, res in enumerate(results, 1):
        print(f"Rank {rank} | Score: {res['score']:.4f}")
        print(f"Title: {res['doc'].get('title', 'No Title')}")
        print(f"Snippet: {res['doc'].get('snippet', '')[:100]}...")
        print("-" * 50)
