import gc
import os
import json
import torch
from tqdm import tqdm
from openai import OpenAI
from transformers import pipeline
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
from retrieve_context_doc import HybridRetriever

class EventCauseRanker(ABC):
    def __init__(self,
                 retriever: HybridRetriever,
                 embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the EventCauseRanker.

        Args:
            retriever: HybridRetriever instance for document retrieval.
            embedding_model_name: Name of the sentence transformer model to use for embedding.
        """
        self.retriever = retriever
        self.embedding_model = SentenceTransformer(embedding_model_name)

    @abstractmethod
    def _call_llm(self, messages_batch) -> list[str]:
        """Call the LLM with a batch of messages and return the list of response texts."""
        raise NotImplementedError
    
    @abstractmethod
    def _clear_llm(self):
        """Clear resources used by the LLM."""
        raise NotImplementedError
    
    def _get_context(self, target_event, topic_id):
        """Retrieves documents for a given event and topic."""
        if not self.retriever:
            raise ValueError("Retriever not initialized")

        return self.retriever.retrieve(target_event, topic_id)

    def _generate_cause_batch(self, target_events, contexts_list):
        """Generates plausible causes for a batch of target events."""
        batch_messages = []
        for target_event, context_docs in zip(target_events, contexts_list):
            context_parts = []
            for res in context_docs:
                title = res.get('title', 'Untitled')
                content = res.get('content', res.get('snippet', ''))
                context_parts.append(f"- [{title}] {content}")

            context_str = "\n".join(context_parts)

            # Concat retrieved documents to beginning of prompt
            prompt = (
                f"Context:\n{context_str}\n\n"
                f"Given the event: '{target_event}', what is a plausible cause? Answer concisely in one sentence."
            )
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant assisting in causal reasoning."},
                {"role": "user", "content": prompt}
            ]
            batch_messages.append(messages)

        return self._call_llm(batch_messages)

    def _calculate_similarity(self, event, options):
        """Calculates cosine similarity between event (or generated cause) and options."""
        option_keys = list(options.keys())
        option_texts = [options[k] for k in option_keys]
        
        target_emb = self.embedding_model.encode([event])
        option_embs = self.embedding_model.encode(option_texts)
        
        similarities = self.embedding_model.similarity(target_emb, option_embs)[0]
        return {k: float(v) for k, v in zip(option_keys, similarities)}

    def _predict_answer(self, similarities):
        """Predicts answer based on highest similarity."""
        return max(similarities, key=similarities.get)
    
    def _get_lines_to_process(self, input_file, output_file):
        processed_ids = set()
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        if 'id' in record:
                            processed_ids.add(record['id'])
                    except json.JSONDecodeError:
                        continue
            print(f"Found {len(processed_ids)} already processed documents.")

        with open(input_file, 'r') as f:
            lines = f.readlines()

        lines_to_process = []
        for line in lines:
            try:
                data = json.loads(line)
                if data.get('id') not in processed_ids:
                    lines_to_process.append(data)
            except json.JSONDecodeError:
                continue

        return lines_to_process
    
    def _run_retrieval_pass(self, input_file, output_file):
        """
        Step 1: Retrieval.
        Reads input_file, retrieves context, and writes to output_file.
        """
        print("Starting Step 1: Retrieval...")
        lines_to_process = self._get_lines_to_process(input_file, output_file)

        if not lines_to_process:
            print("No new documents to process.")
            return

        with open(output_file, 'a') as f_out:
            for data in tqdm(lines_to_process, desc="Retrieving context"):
                target_event = data.get('target_event')
                topic_id = data.get('topic_id')

                context_docs = self._get_context(target_event, topic_id)
                data['context_docs'] = context_docs
                f_out.write(json.dumps(data) + "\n")
                f_out.flush()

    def _run_generation_pass(self, input_file, output_file, batch_size=8):
        """
        Step 2: Generation.
        Reads input_file (with contexts), generates causes, calculates similarity, and writes to output_file.
        """
        print("Starting Step 2: Generation...")
        lines_to_process = self._get_lines_to_process(input_file, output_file)

        if not lines_to_process:
            print("No new documents to process.")
            return
        
        with open(output_file, 'a') as f_out:
            for batch_idx in tqdm(range(0, len(lines_to_process), batch_size), desc="Generating batches"):
                batch_data = lines_to_process[batch_idx : batch_idx + batch_size]

                target_events = []
                contexts_list = []
                options_list = []
                valid_indices = []

                for idx, data in enumerate(batch_data):
                    target_event = data.get('target_event')
                    context_docs = data.get('context_docs')
                    options = {k: v for k, v in data.items() if k.startswith('option_')}

                    target_events.append(target_event)
                    contexts_list.append(context_docs)
                    options_list.append(options)
                    valid_indices.append(idx)

                with torch.no_grad():
                    generated_causes = self._generate_cause_batch(target_events, contexts_list)

                for idx, gen_cause in enumerate(generated_causes):
                    valid_idx = valid_indices[idx]
                    data_item = batch_data[valid_idx]

                    result = data_item.copy()
                    result['generated_cause'] = gen_cause

                    f_out.write(json.dumps(result) + "\n")
                
            f_out.flush() # Ensure all data is written
    
    def _run_similarity_pass(self, input_file, output_file):
        """
        Step 3: Similarity.
        Reads input_file (with contexts and generated causes), calculates similarity, and writes to output_file.
        """
        print("Starting Step 3: Similarity...")
        lines_to_process = self._get_lines_to_process(input_file, output_file)

        if not lines_to_process:
            print("No new documents to process.")
            return
        
        with open(output_file, 'a') as f_out:
            for data in tqdm(lines_to_process, desc="Calculating similarity"):
                generated_cause = data.get('generated_cause')
                options = {k: v for k, v in data.items() if k.startswith('option_')}

                similarities = self._calculate_similarity(generated_cause, options)

                data['similarity'] = similarities
                f_out.write(json.dumps(data) + "\n")
            
            f_out.flush() # Ensure all data is written

    def _run_prediction_pass(self, input_file, output_file):
        """
        Step 4: Prediction.
        Reads input_file (with similarity), and writes to output_file.
        """
        print("Starting Step 4: Prediction...")
        lines_to_process = self._get_lines_to_process(input_file, output_file)

        if not lines_to_process:
            print("No new documents to process.")
            return
        
        with open(output_file, 'a') as f_out:
            for data in tqdm(lines_to_process, desc="Predicting answers"):
                answer = self._predict_answer(data.get('similarity'))
                data['predicted_answer'] = answer
                
                f_out.write(json.dumps(data) + "\n")
            
            f_out.flush()

    def process_file(self, input_file, output_file, batch_size=8):
        """Orchestrates the two-step process: Retrieval -> Release Memory -> Generation -> Release Memory -> Similarity -> Prediction."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Step 1: Retrieval
        context_output_file = output_file.split(".jsonl")[0] + ".context.jsonl"
        self._run_retrieval_pass(input_file, context_output_file)
        
        # Release Retrieval Resources
        print("Releasing retriever resources...")
        self.retriever = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # # Step 2: Generation
        # generation_output_file = output_file.split(".jsonl")[0] + ".generation.jsonl"
        # self._run_generation_pass(context_output_file, generation_output_file, batch_size=batch_size)
        
        # # Release Generation Resources
        # print("Releasing generation resources...")
        # self._clear_llm()
        
        # # Step 3: Similarity
        # similarity_output_file = output_file.split(".jsonl")[0] + ".similarity.jsonl"
        # self._run_similarity_pass(generation_output_file, similarity_output_file)

        # # Release Similarity Resources
        # print("Releasing similarity resources...")
        # self.embedding_model = None
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
        # gc.collect()

        # # Step 4: Prediction
        # self._run_prediction_pass(similarity_output_file, output_file)


class LocalEventCauseRanker(EventCauseRanker):
    def __init__(self, retriever, embedding_model_name="all-MiniLM-L6-v2", model="openai/gpt-oss-safeguard-20b"):
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.pipe = pipeline(
            "text-generation",
            model=model,
            device=device,
            max_new_tokens=2048,
            temperature=0.6, top_p=0.95, top_k=20, min_p=0,
        )
        self.pipe.tokenizer.pad_token_id = self.pipe.model.config.eos_token_id
        super().__init__(retriever, embedding_model_name)

    def _call_llm(self, messages_batch):
        if not self.pipe:
            raise ValueError("Text generation pipeline not initialized or cleared.")

        # Retrieve batch size from the input list length
        outputs = self.pipe(messages_batch, batch_size=len(messages_batch))
        
        results = []
        for output in outputs:
            # Output is a list of dicts (for each sequence generated, usually 1)
            text = output[0]["generated_text"][-1]["content"].strip()

            if "<think>" in text and "</think>" in text:
                non_think_text = text.split("</think>")[1]
                results.append(non_think_text.strip())
            else:
                raise ValueError("No think tag found in LLM output")
                
        return results
    
    def _clear_llm(self):
        self.pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


class LocalServerCauseRanker(EventCauseRanker):
    def __init__(self, retriever,
                 embedding_model_name="all-MiniLM-L6-v2", 
                 model="Qwen/Qwen3-0.6B-GGUF:Q8_0", 
                 base_url="http://localhost:8080/v1", 
                 api_key="no-key"):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.model = model
        super().__init__(retriever, embedding_model_name)

    def _call_llm(self, messages_batch):
        if not self.client:
            raise ValueError("Client not initialized or cleared.")

        results = []
        for messages in messages_batch:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.6,
                    top_p=0.95,
                )
                content = response.choices[0].message.content
                results.append(content)
            except Exception as e:
                print(f"Error calling LLM: {e}")
                results.append("") # Handle error by appending empty string or suitable fallback

        return results

    def _clear_llm(self):
        self.client = None

if __name__ == "__main__":
    docs_path = "semeval2026-task12-dataset/train_data/docs.json"
    retriever = HybridRetriever(docs_path)
    ranker = LocalServerCauseRanker(retriever=retriever)
    ranker.process_file(
        input_file="semeval2026-task12-dataset/train_data/questions.jsonl",
        output_file="results/train_data/output.jsonl",
        batch_size=1
    )