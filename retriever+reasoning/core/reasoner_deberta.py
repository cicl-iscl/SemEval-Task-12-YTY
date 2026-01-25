import json
import os
import torch
from sentence_transformers import CrossEncoder
from tqdm import tqdm
import argparse

class CausalReasonerV2:
    def __init__(self, model_name='microsoft/deberta-v3-large'):
        print(f"Initializing Reasoner V2 with model: {model_name}")
        self.device = 'cpu'
        if torch.backends.mps.is_available():
            self.device = 'mps'
        elif torch.cuda.is_available():
            self.device = 'cuda'
            
        print(f"Using device: {self.device}")
        # Note: deberta-v3-large usually requires a specialized CrossEncoder setup if not using NLI weights,
        # but for simple inference, we can use the base model or a fine-tuned one.
        # ms-marco-MiniLM-L-6-v2 was used before. Let's stick to CrossEncoder interface.
        self.model = CrossEncoder(model_name, device=self.device, max_length=512)
        
    def load_enriched_questions(self, filepath):
        questions = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                questions.append(json.loads(line))
        return questions

    def _prepare_context(self, retrieved_docs):
        """
        Exact same context preparation as used in training.
        """
        return " ".join([f"{d['title']}: {d['content'][:300]}" for d in retrieved_docs[:8]])

    def predict(self, questions, threshold=0.5):
        print(f"Predicting causal relationships (threshold={threshold})...")
        results = []
        
        for q in tqdm(questions):
            target = q['target_event']
            retrieved_docs = q.get('retrieved_docs', [])
            
            # Simple context preparation matching training
            context = self._prepare_context(retrieved_docs)
            
            pairs = []
            options = ['A', 'B', 'C', 'D']
            for opt_key in options:
                option_text = q[f'option_{opt_key}']
                # Use '|' instead of '[SEP]' to match training format
                pairs.append([f"{target} | {option_text}", context])
            
            scores = self.model.predict(pairs)
            
            predicted_answers = []
            option_scores = {}
            for i, opt_key in enumerate(options):
                score = float(scores[i])
                option_scores[opt_key] = score
                if score >= threshold:
                    predicted_answers.append(opt_key)
            
            if not predicted_answers:
                best_idx = scores.argmax()
                predicted_answers.append(options[best_idx])
            
            if len(predicted_answers) == 4:
                min_opt = min(option_scores, key=option_scores.get)
                predicted_answers.remove(min_opt)
                
            q_result = q.copy()
            q_result['inference_scores'] = option_scores
            q_result['predicted_answers'] = predicted_answers
            results.append(q_result)
            
        return results

    def evaluate(self, results):
        # Keep same evaluation logic as V1 for comparability
        total_score = 0
        correct_exact = 0
        total = len(results)
        
        y_true = []
        y_pred = []
        
        for r in results:
            golden = set(r['golden_answer'].split(','))
            predicted = set(r['predicted_answers'])
            
            if predicted == golden:
                instance_score = 1.0
                correct_exact += 1
            elif predicted and predicted.issubset(golden):
                instance_score = 0.5
            else:
                instance_score = 0.0
            total_score += instance_score
                
            for opt in ['A', 'B', 'C', 'D']:
                y_true.append(1 if opt in golden else 0)
                y_pred.append(1 if opt in predicted else 0)
        
        return {
            'official_score': total_score / total,
            'exact_match_accuracy': correct_exact / total,
            'correct_count': correct_exact,
            'total_count': total
        }

def main():
    parser = argparse.ArgumentParser(description="V2 Causal Reasoning with DeBERTa and Hybrid Context.")
    parser.add_argument("--input", type=str, default='dev_data/hybrid_docs.jsonl', help="Path to enriched questions")
    parser.add_argument("--model", type=str, default='causal_fine_tuned_model_phase3', help="HF Model name/path")
    parser.add_argument("--output", type=str, default='reasoning_results_v2.json', help="Output file")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for causal prediction")
    
    args = parser.parse_args()
    
    reasoner = CausalReasonerV2(model_name=args.model)
    questions = reasoner.load_enriched_questions(args.input)
    results = reasoner.predict(questions, threshold=args.threshold)
    
    if questions and 'golden_answer' in questions[0]:
        metrics = reasoner.evaluate(results)
        print(f"\nOfficial Score: {metrics['official_score']:.4f}")
        print(f"EM Accuracy: {metrics['exact_match_accuracy']:.4f}")
    
    # Save (handle both dev analysis and submission format)
    is_dev = questions and 'golden_answer' in questions[0]
    with open(args.output, 'w', encoding='utf-8') as f:
        if is_dev:
            json.dump({'metrics': metrics, 'samples': results}, f, indent=2, ensure_ascii=False)
        else:
            for r in results:
                f.write(json.dumps({"id": r['id'], "answer": ",".join(r['predicted_answers'])}) + '\n')
                
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
