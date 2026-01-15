import json
import os
import torch
from sentence_transformers import CrossEncoder
from tqdm import tqdm
import argparse

class CausalReasoner:
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        print(f"Initializing Reasoner with model: {model_name}")
        self.device = 'cpu'
        if torch.backends.mps.is_available():
            self.device = 'mps'
        elif torch.cuda.is_available():
            self.device = 'cuda'
            
        print(f"Using device: {self.device}")
        self.model = CrossEncoder(model_name, device=self.device)
        
    def load_enriched_questions(self, filepath):
        questions = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                questions.append(json.loads(line))
        return questions

    def predict(self, questions, threshold=0.5):
        print(f"Predicting causal relationships (threshold={threshold})...")
        results = []
        
        for q in tqdm(questions):
            target = q['target_event']
            retrieved_docs = q.get('retrieved_docs', [])
            
            # Combine context from top documents
            # Limit context to avoid exceeding token limits (usually 512)
            context = " ".join([f"{d['title']}: {d['content'][:800]}..." for d in retrieved_docs])
            
            pairs = []
            options = ['option_A', 'option_B', 'option_C', 'option_D']
            for opt_key in options:
                option_text = q[opt_key]
                # Format: "Is [Option] a cause of [Target]? Context: [Context]"
                # Or just use the model's preferred format for Cross-Encoders
                pairs.append([f"{target} | {option_text}", context])
            
            scores = self.model.predict(pairs)
            
            predicted_answers = []
            option_scores = {}
            for i, opt_key in enumerate(['A', 'B', 'C', 'D']):
                score = float(scores[i])
                option_scores[opt_key] = score
                if score >= threshold:
                    predicted_answers.append(opt_key)
            
            # If no answer passes threshold, maybe take the top one?
            # For now, let's keep it strictly threshold-based or top-1 if empty
            if not predicted_answers:
                best_opt = ['A', 'B', 'C', 'D'][scores.argmax()]
                predicted_answers.append(best_opt)
                
            q_result = q.copy()
            q_result['inference_scores'] = option_scores
            q_result['predicted_answers'] = predicted_answers
            results.append(q_result)
            
        return results

    def evaluate(self, results):
        correct_exact = 0
        total = len(results)
        
        y_true = []
        y_pred = []
        
        for r in results:
            golden = set(r['golden_answer'].split(','))
            predicted = set(r['predicted_answers'])
            
            if golden == predicted:
                correct_exact += 1
                
            for opt in ['A', 'B', 'C', 'D']:
                y_true.append(1 if opt in golden else 0)
                y_pred.append(1 if opt in predicted else 0)
        
        accuracy = correct_exact / total
        
        # Calculate P, R, F1 manually to avoid sklearn dependency issues if any
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'exact_match_accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'correct_count': correct_exact,
            'total_count': total
        }

def main():
    parser = argparse.ArgumentParser(description="Reason about causal relationships using a fine-tuned Cross-Encoder.")
    parser.add_argument("--input", type=str, default='dev_data/questions_with_docs.jsonl', help="Path to enriched questions")
    parser.add_argument("--model", type=str, default='causal_fine_tuned_model', help="Path to fine-tuned model")
    parser.add_argument("--output", type=str, default='reasoning_results.json', help="Path to save results")
    parser.add_argument("--threshold", type=float, default=0.0, help="Confidence threshold for causal prediction")
    
    args = parser.parse_args()
    
    model_path = args.model
    if not os.path.exists(model_path):
        print(f"Warning: {model_path} not found. Using default pre-trained model.")
        model_path = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
        
    reasoner = CausalReasoner(model_name=model_path)
    
    questions = reasoner.load_enriched_questions(args.input)
    results = reasoner.predict(questions, threshold=args.threshold)
    
    # Check if we should evaluate (only if golden_answer exists)
    if questions and 'golden_answer' in questions[0]:
        metrics = reasoner.evaluate(results)
        print("\n" + "="*60)
        print("REASONING EVALUATION RESULTS")
        print("="*60)
        print(f"Exact Match Accuracy: {metrics['exact_match_accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Correct: {metrics['correct_count']}/{metrics['total_count']}")
    else:
        print("\nNote: 'golden_answer' not found in data. Skipping evaluation, only predicting.")
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        # If it's test data (no golden), we might want to save just the predictions in the required format
        if not (questions and 'golden_answer' in questions[0]):
            json.dump([{'id': r['id'], 'predicted_answers': r['predicted_answers']} for r in results], f, indent=2)
        else:
            json.dump({
                'metrics': metrics if 'metrics' in locals() else {},
                'samples': results
            }, f, indent=2, ensure_ascii=False)
        
    print(f"\nSaved results to {args.output}")

if __name__ == "__main__":
    main()
