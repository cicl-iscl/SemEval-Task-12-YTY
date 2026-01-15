import json
import os
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import argparse

class DocRetriever:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print(f"Initializing Retriever with model: {model_name}")
        # Force CPU if needed, but let's try default (should handle M1/Mac)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.backends.mps.is_available():
            self.device = 'mps'
        
        print(f"Using device: {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
        
    def load_data(self, questions_path, docs_path):
        print("Loading data...")
        questions = []
        with open(questions_path, 'r', encoding='utf-8') as f:
            for line in f:
                questions.append(json.loads(line))
        
        with open(docs_path, 'r', encoding='utf-8') as f:
            docs_by_topic = json.load(f)
            
        # Convert list to dict for fast access
        self.topic_map = {item['topic_id']: item['docs'] for item in docs_by_topic}
        return questions

    def retrieve_for_questions(self, questions, top_k=3):
        print(f"Retrieving top {top_k} documents for each question...")
        results = []
        
        # We can group questions by topic to optimize encoding if needed, 
        # but let's keep it simple first.
        for q in tqdm(questions):
            topic_id = q['topic_id']
            target_event = q['target_event']
            
            if topic_id not in self.topic_map:
                results.append(q)
                continue
                
            topic_docs = self.topic_map[topic_id]
            doc_texts = [f"{d.get('title', '')} {d.get('snippet', '')}" for d in topic_docs]
            
            # Encode target and all docs in this topic
            target_emb = self.model.encode(target_event, convert_to_tensor=True)
            doc_embs = self.model.encode(doc_texts, convert_to_tensor=True)
            
            # Compute similarities
            cos_scores = util.cos_sim(target_emb, doc_embs)[0]
            
            # Get top k
            top_results = torch.topk(cos_scores, k=min(top_k, len(doc_texts)))
            
            retrieved_docs = []
            for score, idx in zip(top_results[0], top_results[1]):
                doc = topic_docs[idx]
                retrieved_docs.append({
                    'id': doc['id'],
                    'title': doc['title'],
                    'content': doc['content'],
                    'score': float(score)
                })
            
            q_result = q.copy()
            q_result['retrieved_docs'] = retrieved_docs
            results.append(q_result)
            
        return results

def run_retrieval(questions_path, docs_path, output_path):
    retriever = DocRetriever()
    questions = retriever.load_data(questions_path, docs_path)
    enriched_questions = retriever.retrieve_for_questions(questions, top_k=3)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for q in enriched_questions:
            f.write(json.dumps(q, ensure_ascii=False) + '\n')
            
    print(f"Saved enriched questions to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Retrieve relevant documents for causal reasoning questions.")
    parser.add_argument("--questions", type=str, default='dev_data/questions.jsonl', help="Path to questions.jsonl")
    parser.add_argument("--docs", type=str, default='dev_data/docs.json', help="Path to docs.json")
    parser.add_argument("--output", type=str, default='dev_data/questions_with_docs.jsonl', help="Path to save enriched questions")
    parser.add_argument("--top_k", type=int, default=3, help="Number of documents to retrieve")
    
    args = parser.parse_args()
    
    print(f"Enriching data from {args.questions}...")
    run_retrieval(args.questions, args.docs, args.output)

if __name__ == "__main__":
    main()
