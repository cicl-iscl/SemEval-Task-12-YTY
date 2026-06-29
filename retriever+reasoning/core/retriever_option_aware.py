import json
import os
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import argparse

class DocRetrieverV2:
    def __init__(self, model_name='all-MiniLM-L12-v2'):
        print(f"Initializing Retriever V2 with model: {model_name}")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.backends.mps.is_available():
            self.device = 'mps'
        
        print(f"Using device: {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
    

    def load_data(self, questions_path, docs_path):
        print(f"Loading questions from {questions_path} and docs from {docs_path}...")
        questions = []
        with open(questions_path, 'r', encoding='utf-8') as f:
            for line in f:
                questions.append(json.loads(line))
        
        with open(docs_path, 'r', encoding='utf-8') as f:
            docs_by_topic = json.load(f)
            
        self.topic_map = {item['topic_id']: item['docs'] for item in docs_by_topic}
        return questions

    def retrieve_hybrid_docs(self, questions, global_k=2, option_k=1):
        print(f"Performing Hybrid Retrieval (Global K={global_k}, Option K={option_k})...")
        results = []
        
        for q in tqdm(questions):
            topic_id = q['topic_id']
            target_event = q['target_event']
            options = {
                'A': q['option_A'],
                'B': q['option_B'],
                'C': q['option_C'],
                'D': q['option_D']
            }
            
            if topic_id not in self.topic_map:
                q['retrieved_docs'] = []
                results.append(q)
                continue
                
            topic_docs = self.topic_map[topic_id]
            # Use title + snippet for embedding search
            doc_texts = [f"{doc.get('title', '')} {doc.get('snippet', '')}" for doc in topic_docs]
            doc_embs = self.model.encode(doc_texts, convert_to_tensor=True)
            
            retrieved_ids = set()
            final_docs = []
            
            # Helper to add top docs from scores
            def add_top_docs(scores, k, source_type):
                top_results = torch.topk(scores, k=min(k, len(doc_texts)))
                for score, idx in zip(top_results[0], top_results[1]):
                    doc = topic_docs[idx]
                    if doc['id'] not in retrieved_ids:
                        retrieved_ids.add(doc['id'])
                        final_docs.append({
                            'id': doc['id'],
                            'title': doc['title'],
                            'content': doc['content'],
                            'score': float(score), 
                            'source': source_type
                        })

            # 1. Global Search: Target Event only
            target_emb = self.model.encode(target_event, convert_to_tensor=True)
            target_scores = util.cos_sim(target_emb, doc_embs)[0]
            add_top_docs(target_scores, global_k, 'global')
            
            # 2. Option-Aware Search: Target + each Option
            for opt_key, opt_text in options.items():
                query = f"{target_event} {opt_text}"
                query_emb = self.model.encode(query, convert_to_tensor=True)
                query_scores = util.cos_sim(query_emb, doc_embs)[0]
                add_top_docs(query_scores, option_k, f'option_{opt_key}')
            
            q_result = q.copy()
            # Sort by score for consistency, but keep record of source
            q_result['retrieved_docs'] = sorted(final_docs, key=lambda x: x['score'], reverse=True)
            results.append(q_result)
            
        return results

def main():
    parser = argparse.ArgumentParser(description="Hybrid Retrieval (Global + Option-Aware) for causal reasoning.")
    parser.add_argument("--questions", type=str, default='dev_data/questions.jsonl', help="Path to questions.jsonl")
    parser.add_argument("--docs", type=str, default='dev_data/docs.json', help="Path to docs.json")
    parser.add_argument("--output", type=str, default='dev_data/hybrid_docs.jsonl', help="Path to save enriched questions")
    parser.add_argument("--global_k", type=int, default=2, help="Global docs to retrieve")
    parser.add_argument("--option_k", type=int, default=1, help="Docs per option to retrieve")
    
    args = parser.parse_args()
    
    retriever = DocRetrieverV2()
    questions = retriever.load_data(args.questions, args.docs)
    enriched = retriever.retrieve_hybrid_docs(questions, global_k=args.global_k, option_k=args.option_k)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        for q in enriched:
            f.write(json.dumps(q, ensure_ascii=False) + '\n')
            
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
