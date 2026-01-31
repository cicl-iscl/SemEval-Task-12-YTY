import os
import json
import torch
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, CrossEncoder

class HybridRetriever:
    def __init__(self, data_path: str,
                 dense_model_name='all-MiniLM-L6-v2',
                 rerank_model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        """
        Initializes the retriever with the specified models and data.
        """
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")

        print(f"Loading dense model: {dense_model_name}...")
        self.dense_model = SentenceTransformer(dense_model_name, device=self.device)

        print(f"Loading reranker model: {rerank_model_name}...")
        self.reranker = CrossEncoder(rerank_model_name, device=self.device)

        print(f"Loading data from {data_path}...")
        self.documents = self._load_data(data_path)

        self._dense_index = {}
        self._sparse_index = {}

    def _load_data(self, data_path: str):
        """
        Loads documents from the specified JSON file.
        The expected format is:
        {
            "topic_id": ...
            "docs": [
                {"id": "...", "content": "...", ...},
                ...
            ]
        }
        """
        # Check if file exists
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"File not found: {data_path}")

        # Load data
        with open(data_path, 'r') as f:
            data = json.load(f)

        return {topic['topic_id']: topic['docs'] for topic in data}

    def _get_dense_index(self, topic_id: int) -> faiss.IndexFlatIP:
        """
        Returns the dense index for the given topic ID.
        If the index is not already built, it builds it and caches it.
        """
        if topic_id not in self._dense_index:
            corpus_embeddings = self.dense_model.encode([doc['content'] for doc in self.documents[topic_id]], show_progress_bar=False)
            faiss.normalize_L2(corpus_embeddings) # Normalize embeddings for cosine similarity
            dense_index = faiss.IndexFlatIP(corpus_embeddings.shape[1]) # Inner Product (sim to Cosine after normalization)
            dense_index.add(corpus_embeddings)
            self._dense_index[topic_id] = dense_index

        return self._dense_index[topic_id]

    def _get_sparse_index(self, topic_id: int) -> BM25Okapi:
        """
        Returns the sparse index for the given topic ID.
        If the index is not already built, it builds it and caches it.
        """
        if topic_id not in self._sparse_index:
            tokenized_corpus = [doc['content'].split() for doc in self.documents[topic_id]]
            sparse_index = BM25Okapi(tokenized_corpus)
            self._sparse_index[topic_id] = sparse_index

        return self._sparse_index[topic_id]

    def retrieve(self,
                 query: str,
                 topic_id: int,
                 top_k_dense=20,
                 top_k_sparse=20,
                 top_k_final=5) -> List[Dict[str, Any]]:
        """
        Returns:
            List of retrieved documents.
            Each document is a dictionary with the following keys:
                - id: The document ID.
                - content: The document content.
                - score: The reranking score.
        """
        # 1. Dense Retrieval
        query_embedding = self.dense_model.encode([query])
        faiss.normalize_L2(query_embedding)
        dense_index = self._get_dense_index(topic_id)
        _, dense_indices = dense_index.search(query_embedding, top_k_dense)

        # 2. Sparse Retrieval
        tokenized_query = query.split()
        sparse_index = self._get_sparse_index(topic_id)
        sparse_scores = sparse_index.get_scores(tokenized_query)
        sparse_indices = np.argsort(sparse_scores)[::-1][:top_k_sparse]

        # 3. Combine Candidates
        candidate_indices = list(set(dense_indices[0]).union(set(sparse_indices)))

        # 4. Reranking
        candidates = [self.documents[topic_id][i] for i in candidate_indices]
        sentence_pairs = [[query, doc["content"]] for doc in candidates] # Query + Document Full Content
        rerank_scores = self.reranker.predict(sentence_pairs)

        # Associate scores with candidates
        results_with_scores = []
        for i, score in enumerate(rerank_scores):
            results_with_scores.append({
                "doc": candidates[i],
                "score": score
            })

        # Sort by rerank score descending
        results_with_scores.sort(key=lambda x: x["score"], reverse=True)

        return [res["doc"] for res in results_with_scores[:top_k_final]]

if __name__ == "__main__":
    # Example Usage
    dataset_path = "semeval2026-task12-dataset/dev_data/docs.json"
    retriever = HybridRetriever(dataset_path)
    
    # Sample Query
    query = "President Yoon Suk Yeol vowed to carry out a thorough investigation."
    print(f"\nQuery: {query}")
    topic_id = 7
    results = retriever.retrieve(query, topic_id)
    
    for rank, res in enumerate(results, 1):
        print(f"Rank {rank} | Score: {res['score']:.4f}")
        print(f"Title: {res['doc'].get('title', 'No Title')}")
        print(f"Snippet: {res['doc'].get('snippet', '')[:100]}...")
        print("-" * 50)
