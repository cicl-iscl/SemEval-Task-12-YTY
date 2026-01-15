import json
import torch
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

class CausalFineTuner:
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        print(f"Initializing Fine-Tuner with model: {model_name}")
        self.device = 'cpu'
        if torch.backends.mps.is_available():
            self.device = 'mps'
        elif torch.cuda.is_available():
            self.device = 'cuda'
        
        print(f"Using device: {self.device}")
        self.model = CrossEncoder(model_name, num_labels=1, device=self.device)
        
    def load_data(self, filepath):
        print(f"Loading enriched data from {filepath}...")
        examples = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                q = json.loads(line)
                target = q['target_event']
                golden = set(q['golden_answer'].split(','))
                retrieved_docs = q.get('retrieved_docs', [])
                context = " ".join([f"{d['title']}: {d['content'][:500]}" for d in retrieved_docs])
                
                for opt_key in ['A', 'B', 'C', 'D']:
                    opt_text = q[f'option_{opt_key}']
                    label = 1.0 if opt_key in golden else 0.0
                    # Input format: [Target | Option, Context]
                    examples.append(InputExample(texts=[f"{target} | {opt_text}", context], label=label))
        
        print(f"Loaded {len(examples)} training examples.")
        return examples

    def train(self, train_examples, output_path, epochs=3, batch_size=16):
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        
        print(f"Starting fine-tuning for {epochs} epochs...")
        # CrossEncoder.fit handles the training loop
        self.model.fit(
            train_dataloader=train_dataloader,
            epochs=epochs,
            warmup_steps=100,
            output_path=output_path,
            show_progress_bar=True
        )
        # Explicit save to ensure it persists correctly
        self.model.save(output_path)
        print(f"Training complete. Model saved to {output_path}")

def main():
    fine_tuner = CausalFineTuner()
    
    train_path = 'train_data/questions_with_docs.jsonl'
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found. Run retriever.py first.")
        return
        
    train_examples = fine_tuner.load_data(train_path)
    
    output_model_path = 'causal_fine_tuned_model'
    fine_tuner.train(train_examples, output_model_path, epochs=3, batch_size=8)

if __name__ == "__main__":
    main()
