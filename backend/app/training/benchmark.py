import argparse
import json
import os
import random
import re
import sys
import time
import requests
import torch
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
from tqdm import tqdm

def evaluate_response(response_text: str, ground_truth_tags: list):
    """
    Evaluates response with support for Dict format {"discovered_techniques": []}
    and Markdown stripping.
    """
    parsed_tags = []
    parsing_status = 'Failed'

    # 0. Pre-processing: Strip Markdown (Crucial for Strict Success)
    clean_text = response_text.replace("```json", "").replace("```", "").strip()

    # Attempt 1: Strict JSON parsing
    try:
        parsed_output = json.loads(clean_text)

        # CASE A: Output is the expected Dictionary
        if isinstance(parsed_output, dict):
            # Extract the specific key we trained on
            parsed_tags = parsed_output.get("discovered_techniques", [])
            # Check if the inner content is actually a list
            if not isinstance(parsed_tags, list):
                 # Try to force it if it's a string representation
                 parsed_tags = []
            parsing_status = 'Strict Success'

        # CASE B: Model outputted a raw List (unlikely but possible)
        elif isinstance(parsed_output, list):
            parsed_tags = parsed_output
            parsing_status = 'Strict Success'

        else:
            raise ValueError("Parsed output is not a Dict or List.")

    except (json.JSONDecodeError, ValueError):
        # Attempt 2: Regex-based correction
        # We look for the list explicitly
        match = re.search(r'\[(.*?)\]', clean_text, re.DOTALL)
        if match:
            extracted_content = f"[{match.group(1)}]"
            try:
                parsed_output_recovered = json.loads(extracted_content)
                if isinstance(parsed_output_recovered, list):
                    parsed_tags = parsed_output_recovered
                    parsing_status = 'Recovered'
            except (json.JSONDecodeError, ValueError):
                pass

    # Clean tags and convert to sets for easier set operations
    parsed_tags_set = set(str(tag) for tag in parsed_tags if tag is not None)
    ground_truth_tags_set = set(str(tag) for tag in ground_truth_tags if tag is not None)

    # --- Document-level F1 calculation (as per user definition) ---
    # TP = |pred ∩ gold|
    tp_doc = len(parsed_tags_set.intersection(ground_truth_tags_set))
    # FP = |pred − gold|
    fp_doc = len(parsed_tags_set.difference(ground_truth_tags_set))
    # FN = |gold − pred|
    fn_doc = len(ground_truth_tags_set.difference(parsed_tags_set))

    # F1_doc = 0 if TP=FP=FN=0, else 2*TP / (2*TP + FP + FN)
    if tp_doc == 0 and fp_doc == 0 and fn_doc == 0:
        f1_doc = 0.0 # Per user instruction for when both sets are empty
    else:
        f1_doc = (2 * tp_doc) / (2 * tp_doc + fp_doc + fn_doc)

    # Exact-match accuracy
    exact_match = (parsed_tags_set == ground_truth_tags_set)

    return {
        'parsing_status': parsing_status,
        'parsed_tags': list(parsed_tags_set), # Store as list for consistency
        'f1_doc': f1_doc,
        'exact_match': exact_match,
        'has_gold_labels': bool(ground_truth_tags_set), # To identify documents with non-empty gold labels
        'ground_truth': list(ground_truth_tags_set),
        'predicted': list(parsed_tags_set),
        'raw_output': response_text
    }

def format_prompt(example, tokenizer):
    # Combine instruction for system message and input for the user message
    system_instruction = '''
Jesteś ekspertem w dziedzinie analizy mediów i lingwistyki, specjalizującym się w wykrywaniu propagandy, manipulacji poznawczej i błędów logicznych w tekstach w języku polskim.

**Twoje zadanie:**
Przeanalizuj dostarczony tekst wejściowy w języku polskim, aby zidentyfikować konkretne techniki manipulacji. Musisz oprzeć swoją analizę wyłącznie na dostarczonym tekście, szukając wzorców, które mają na celu wpłynięcie na opinię czytelnika za pomocą środków irracjonalnych lub zwodniczych.

**Dozwolone kategorie manipulacji:**
Jesteś ściśle ograniczony do klasyfikowania technik w następujących kategoriach. Nie używaj żadnych innych tagów.

1.  **REFERENCE_ERROR**: Cytaty, które nie popierają tezy, są zmyślone lub pochodzą z niewiarygodnych źródeł.
2.  **WHATABOUTISM**: Dyskredytowanie stanowiska oponenta poprzez zarzucanie mu hipokryzji, bez bezpośredniego odparcia jego argumentów.
3.  **STRAWMAN**: Przeinaczenie argumentu oponenta (stworzenie "chochoła"), aby łatwiej go było zaatakować.
4.  **EMOTIONAL_CONTENT**: Używanie języka nasyconego emocjami (strach, gniew, litość, radość) w celu ominięcia racjonalnego, krytycznego myślenia.
5.  **CHERRY_PICKING**: Zatajanie dowodów lub ignorowanie danych, które zaprzeczają argumentowi, przy jednoczesnym przedstawianiu tylko danych potwierdzających.
6.  **FALSE_CAUSE**: Błędne zidentyfikowanie przyczyny zjawiska (np. mylenie korelacji z przyczynowością).
7.  **MISLEADING_CLICKBAIT**: Nagłówki lub wstępy, które sensacyjnie wyolbrzymiają lub fałszywie przedstawiają faktyczną treść tekstu.
8.  **ANECDOTE**: Wykorzystywanie odosobnionych historii osobistych lub pojedynczych przykładów jako ważnego dowodu na ogólny trend lub fakt naukowy.
9.  **LEADING_QUESTIONS**: Pytania sformułowane w sposób sugerujący konkretną odpowiedź lub zawierające nieudowodnione założenie.
10. **EXAGGERATION**: Hiperboliczne stwierdzenia, które wyolbrzymiają fakty, aby wywołać reakcję.
11. **QUOTE_MINING**: Wyrywanie cytatów z kontekstu w celu zniekształcenia intencji pierwotnego autora.

**Format wyjściowy:**
Musisz odpowiedzieć pojedynczym, poprawnym obiektem JSON zawierającym dwa klucze:
1.  `"reasoning"`: Spójny akapit w **języku polskim** wyjaśniający, które techniki znaleziono i dlaczego. Musisz przytoczyć konkretną logikę lub fragmenty tekstu, aby uzasadnić swoją klasyfikację.
2.  `"discovered_techniques"`: Lista ciągów znaków (stringów) zawierająca dokładnie te tagi, które zdefiniowano powyżej. Jeśli nie znaleziono żadnych technik, zwróć pustą listę.

**Przykładowa struktura:**
{
    "reasoning": "Tekst stosuje [Nazwa Techniki], ponieważ autor sugeruje, że...",
    "discovered_techniques": ["NAZWA_TECHNIKI"]
}
    '''
    user_message = example['input']

    # Construct the ChatML formatted prompt
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_message},
    ]
    # We don't add generation prompt here because unsloth handles it or we do it manually? 
    # Notebook says: add_generation_prompt=True
    example['prompt'] = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Parse tags from output for ground truth
    try:
        clean_json = example['output'].replace("```json", "").replace("```", "").strip()
        example['tags'] = json.loads(clean_json)['discovered_techniques']
    except Exception:
        example['tags'] = []
    
    return example

def report_progress(url, value):
    try:
        requests.post(f"{url}/training/progress", 
                      json={"stage": "evaluation", "value": value}, 
                      timeout=1)
    except:
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, required=True, help="Path to adapter")
    parser.add_argument("--base", type=str, required=True, help="Path to base model")
    parser.add_argument("--data", type=str, required=True, help="Path to test dataset (.jsonl)")
    parser.add_argument("--backend", type=str, default="http://localhost:8000", help="Backend URL")
    parser.add_argument("--output_dir", type=str, default="./model/benchmark_reports", help="Output directory for reports")
    parser.add_argument("--no-tqdm", action="store_true", help="Disable tqdm progress bar")
    
    args = parser.parse_args()

    print(f"DEBUG: Starting benchmark with adapter={args.adapter}, base={args.base}")
    
    # 1. Load Model
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.adapter, # Load adapter directly (unsloth supports this)
        max_seq_length = 2048, # Adjust as needed
        load_in_4bit = True,
        use_gradient_checkpointing = "unsloth",
    )
    FastLanguageModel.for_inference(model)
    
    # 2. Load Dataset
    print(f"Loading dataset from {args.data}...")
    dataset = load_dataset("json", data_files=args.data, split="train") # It's 'train' split by default for jsonl unless specified
    
    # 3. Sample 5 items
    print("Sampling 5 items...")
    dataset = dataset.shuffle(seed=42)
    sample_size = min(5, len(dataset))
    dataset = dataset.select(range(sample_size))
    
    # 4. Format Prompts
    print("Formatting prompts...")
    dataset = dataset.map(lambda x: format_prompt(x, tokenizer))
    
    # 5. Inference
    print("Running inference...")
    results = []
    
    iterator = dataset if args.no_tqdm else tqdm(dataset)
    for i, example in enumerate(iterator):
        prompt = example['prompt']
        ground_truth = example['tags']
        
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, 
                max_new_tokens=512, 
                use_cache=True,
                temperature=0.0 # Greedy decoding
            )
            
        # Decode only new tokens
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        response_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Evaluate
        eval_result = evaluate_response(response_text, ground_truth)
        results.append(eval_result)
        
        # Determine progress
        progress_val = int((i + 1) / sample_size * 100)
        report_progress(args.backend, progress_val)
        
    # 6. Aggregate Metrics
    total_docs = len(results)
    strict_success_count = sum(1 for r in results if r['parsing_status'] == 'Strict Success')
    recovered_count = sum(1 for r in results if r['parsing_status'] == 'Recovered')
    
    # New metrics logic
    total_f1_doc = sum(r['f1_doc'] for r in results)
    
    total_f1_doc_non_empty_gold = sum(r['f1_doc'] for r in results if r['has_gold_labels'])
    non_empty_gold_docs_count = sum(1 for r in results if r['has_gold_labels'])
    
    exact_matches_count = sum(1 for r in results if r['exact_match'])
    
    # Calculations
    parsing_success_rate = strict_success_count / total_docs if total_docs > 0 else 0
    mean_f1_doc_all_docs = total_f1_doc / total_docs if total_docs > 0 else 0
    mean_f1_doc_non_empty = total_f1_doc_non_empty_gold / non_empty_gold_docs_count if non_empty_gold_docs_count > 0 else 0
    exact_match_accuracy = exact_matches_count / total_docs if total_docs > 0 else 0
    
    print(f"RESULT: Mean Document-Level F1 (excluding empty gold-label docs): {mean_f1_doc_non_empty:.4f}")
    
    # Machine readable tokens for orchestrator
    final_f1_to_report = mean_f1_doc_non_empty if non_empty_gold_docs_count > 0 else 0.0
    print(f"FINAL_F1_SCORE: {final_f1_to_report:.4f}")
    print(f"FINAL_EXACT_MATCH: {exact_match_accuracy:.4f}")
    
    # 7. Generate Report content
    report_lines = []
    report_lines.append("="*60)
    report_lines.append(f"INFERENCE REPORT: {total_docs} documents")
    report_lines.append(f"Adapter: {args.adapter}")
    report_lines.append(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("="*60)
    
    report_lines.append(f"Parsing Success Rate (Strict JSON with structure validation i.e. reasoning is spelled correctly): {parsing_success_rate:.4f} ({strict_success_count}/{total_docs})")
    report_lines.append(f"Exact-Match Accuracy: {exact_match_accuracy:.4f} ({exact_matches_count}/{total_docs})")
    
    if non_empty_gold_docs_count > 0:
        report_lines.append(f"Mean Document-Level F1 (excluding empty gold-label docs): {mean_f1_doc_non_empty:.4f}")
    else:
        report_lines.append("Mean Document-Level F1 (excluding empty gold-label docs): N/A (No documents with gold labels found)")
        
    report_lines.append("-" * 60)
    
    # Write to file
    os.makedirs(args.output_dir, exist_ok=True)
    filename = f"benchmark_report_{int(time.time())}.txt"
    output_path = os.path.join(args.output_dir, filename)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        
    print(f"Report written to: {output_path}")
    # We yield the F1 (excluding empty) as the primary metric for promotion logic if needed, or stick to all docs? 
    # User didn't specify which one drives promotion, but usually F1 (non-empty) is strictly harder and better signal.
    # For compatibility with frontend that expects one number, let's output the strict one (non-empty) or safe fallback.
    final_metric = mean_f1_doc_non_empty if non_empty_gold_docs_count > 0 else mean_f1_doc_all_docs
    print(f"FINAL_F1_SCORE:{final_metric:.4f}")
    
if __name__ == "__main__":
    main()
