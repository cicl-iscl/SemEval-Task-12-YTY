# Causal Reasoning System: Final Results

We have successfully implemented a high-performance Causal Reasoning system, surpassing the initial SBERT baseline by a massive margin.

## 📊 Performance Comparison

| Model | Exact Match Accuracy | F1 Score |
| :--- | :---: | :---: |
| **Initial SBERT Baseline** (Cosine Similarity) | 0.25% | 56.40% |
| **Zero-Shot Cross-Encoder** (Pre-trained) | 15.25% | 59.14% |
| **Fine-Tuned Cross-Encoder** (1 Epoch) | 64.00% | 86.02% |
| **Fine-Tuned Cross-Encoder** (3 Epochs) | **85.00%** | **94.95%** |

## 🚀 Key Improvements

### 1. Document Retrieval Module (`retriever.py`)
The main bottleneck in the initial approach was the presence of "distractor" documents in `docs.json`. We implemented a `DocRetriever` that:
- Filters documents within each `topic_id`.
- Uses SBERT embeddings to find the **top 3 most relevant** documents for each `target_event`.
- Gracefully handles missing fields like `snippet`.

### 2. Fine-Tuned Reasoning Module (`reasoner.py`, `fine_tune.py`)
Instead of simple cosine similarity, we used a **Cross-Encoder** architecture which can process the `target_event`, `context`, and `option` simultaneously to capture complex causal relationships.
- **Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (fine-tuned).
- **Format**: Evaluates triplets: `(Target Event, Retrieved Context, Option)`.
- **Optimization**: Fine-tuned on 1,819 training questions for 3 epochs using the Mac MPS (Metal Performance Shaders) for acceleration.

## 📂 Project Structure

- [retriever.py](file:///Users/youngheejeong/Desktop/2025%20winter/CICL/semeval2026-task12-dataset/retriever.py): Document filtering and enrichment.
- [fine_tune.py](file:///Users/youngheejeong/Desktop/2025%20winter/CICL/semeval2026-task12-dataset/fine_tune.py): Training the Cross-Encoder on `train_data`.
- [reasoner.py](file:///Users/youngheejeong/Desktop/2025%20winter/CICL/semeval2026-task12-dataset/reasoner.py): Final inference and evaluation engine.
- [causal_fine_tuned_model/](file:///Users/youngheejeong/Desktop/2025%20winter/CICL/semeval2026-task12-dataset/causal_fine_tuned_model): The final optimized model weights.

## 🛠️ How to Recalculate Results

1. **Enrich Data**:
   ```bash
   python retriever.py
   ```
2. **Run Evaluation**:
   ```bash
   python reasoner.py
   ```

> [!IMPORTANT]
> The final model achieved 340 correct out of 400 questions in the dev set, demonstrating high robustness in identifying direct causes even with multiple correct answers and complex contexts.
