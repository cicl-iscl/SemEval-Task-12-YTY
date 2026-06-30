import json
import torch
import argparse
import os
import numpy as np
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

class SimpleCEEvaluator:
    # CEBinaryClassificationEvaluator was acting up. 
    # Current Cross-Encoder Evaluator in Transformer library has some dependency issues on Colab and is not iterable.
    # This is a simplified version.
    def __init__(self, sentences1, sentences2, labels, name='dev_set'):
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.labels = labels
        self.name = name

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        # Predict in chunks to save memory
        model_predictions = model.predict(
            list(zip(self.sentences1, self.sentences2)), 
            batch_size=16, 
            show_progress_bar=False
        )
        
        # Binary classification (Threshold 0.5)
        preds = [1 if s > 0.5 else 0 for s in model_predictions]
        acc = accuracy_score(self.labels, preds)
        
        print(f"\n[Evaluation] Step {steps}: Accuracy = {acc:.4f}")

        return acc

    def __iter__(self):
        return iter([self])

class CausalFineTuner:
    def __init__(self, model_name='microsoft/deberta-v3-large'):
        print(f"Model: {model_name}")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Device: {self.device}")
        self.model = CrossEncoder(model_name, num_labels=1, device=self.device)
        
    def load_data(self, filepath, is_dev=False):
        if not os.path.exists(filepath):
            print(f"Error: {filepath} not found.")
            return None
        
        print(f"Loading data from {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
            
        if is_dev:
            return data
        
        examples = []
        for q in data: # Nested loops
            target = q['target_event']
            golden = set(str(q['golden_answer']).split(','))
            retrieved_docs = q.get('retrieved_docs', [])
            
            # Context concatenation (limit to 8 docs)
            context = " ".join([f"{d['title']}: {d['content'][:300]}" for d in retrieved_docs[:8]])
            # Format:
            # {Target} | {Option} <SEP> {Context} ==> each question diverged into 4 single binary classfication problems.
            for opt_key in ['A', 'B', 'C', 'D']:
                opt_text = q[f'option_{opt_key}']
                label = 1.0 if opt_key in golden else 0.0
                examples.append(InputExample(texts=[f"{target} | {opt_text}", context], label=label)) # [s1, s2], l
        
        print(f"Created {len(examples)} training examples.")
        return examples
    
    def create_dev_evaluator(self, dev_questions):
        s1, s2, labels = [], [], []
        for q in dev_questions:
            target = q['target_event']
            golden = set(str(q['golden_answer']).split(','))
            retrieved_docs = q.get('retrieved_docs', [])
            context = " ".join([f"{d['title']}: {d['content'][:300]}" for d in retrieved_docs[:8]])
            
            for opt_key in ['A', 'B', 'C', 'D']:
                s1.append(f"{target} | {q[f'option_{opt_key}']}")
                s2.append(context)
                labels.append(1 if opt_key in golden else 0)
        
        return SimpleCEEvaluator(s1, s2, labels)

    def train(self, train_examples, dev_evaluator, output_path, epochs=3, batch_size=4, lr=5e-6):
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        
        print(f"Starting training on {self.device}...")
        self.model.fit(
            train_dataloader=train_dataloader,
            evaluator=dev_evaluator,
            epochs=epochs,
            warmup_steps=300,
            optimizer_params={'lr': lr},
            output_path=output_path,
            show_progress_bar=True,
            evaluation_steps=500 if dev_evaluator else 0,
            save_best_model=True if dev_evaluator else False,
            use_amp=True  # Mixed precision for T4 GPU
        )
        print(f"Training finished. Best model should be in: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='microsoft/deberta-v3-large')
    parser.add_argument("--train_data", type=str, default='train_data/hybrid_docs.jsonl')
    parser.add_argument("--dev_data", type=str, default='dev_data/hybrid_docs.jsonl')
    parser.add_argument("--output", type=str, default='causal_fine_tuned_model_large')
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    
    args = parser.parse_args()
    
    tuner = CausalFineTuner(model_name=args.model)
    
    # Load data
    train_ex = tuner.load_data(args.train_data, is_dev=False)
    dev_eval = None
    if args.dev_data and os.path.exists(args.dev_data):
        dev_qs = tuner.load_data(args.dev_data, is_dev=True)
        dev_eval = tuner.create_dev_evaluator(dev_qs)
    
    # Run training
    tuner.train(
        train_examples=train_ex,
        dev_evaluator=dev_eval,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )

if __name__ == "__main__":
    main()
