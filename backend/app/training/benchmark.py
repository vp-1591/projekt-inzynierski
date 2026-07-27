import argparse
import contextlib
import json
import os
import re
import time

import requests
import torch
from datasets import load_dataset
from tqdm import tqdm
from unsloth import FastLanguageModel


def evaluate_response(response_text: str, ground_truth_tags: list):
    """
    Evaluates LLM response by comparing predicted tags against ground truth.
    Supports both structured JSON parsing and regex-based recovery.
    """
    parsed_tags = []
    parsing_status = 'Failed'

    # Pre-processing: Clean markdown wrappers
    clean_text = response_text.replace("```json", "").replace("```", "").strip()

    # Phase 1: Structured JSON Parsing
    try:
        parsed_output = json.loads(clean_text)

        if isinstance(parsed_output, dict):
            parsed_tags = parsed_output.get("discovered_techniques", [])
            if not isinstance(parsed_tags, list): parsed_tags = []
            parsing_status = 'Strict Success'

        elif isinstance(parsed_output, list):
            parsed_tags = parsed_output
            parsing_status = 'Strict Success'

    except (json.JSONDecodeError, ValueError):
        # Phase 2: Regex Recovery (finding list within text)
        match = re.search(r'\[(.*?)\]', clean_text, re.DOTALL)
        if match:
            try:
                parsed_output_recovered = json.loads(f"[{match.group(1)}]")
                if isinstance(parsed_output_recovered, list):
                    parsed_tags = parsed_output_recovered
                    parsing_status = 'Recovered'
            except: pass

    # Comparison using sets
    parsed_tags_set = set(str(tag) for tag in parsed_tags if tag is not None)
    ground_truth_tags_set = set(str(tag) for tag in ground_truth_tags if tag is not None)

    # Metric: Document-level F1
    tp_doc = len(parsed_tags_set.intersection(ground_truth_tags_set))
    fp_doc = len(parsed_tags_set.difference(ground_truth_tags_set))
    fn_doc = len(ground_truth_tags_set.difference(parsed_tags_set))

    if tp_doc == 0 and fp_doc == 0 and fn_doc == 0:
        f1_doc = 0.0 # Standardize score for empty-vs-empty matches
    else:
        f1_doc = (2 * tp_doc) / (2 * tp_doc + fp_doc + fn_doc)

    exact_match = (parsed_tags_set == ground_truth_tags_set)

    return {
        'parsing_status': parsing_status,
        'parsed_tags': list(parsed_tags_set),
        'f1_doc': f1_doc,
        'exact_match': exact_match,
        'has_gold_labels': bool(ground_truth_tags_set),
        'ground_truth': list(ground_truth_tags_set),
        'predicted': list(parsed_tags_set),
        'raw_output': response_text
    }

def format_prompt(example, tokenizer):
    """Wraps the input text in the detection system prompt and prepares chat template."""
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
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": example['input']},
    ]
    example['prompt'] = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Extract ground truth tags from the dataset
    try:
        clean_json = example['output'].replace("```json", "").replace("```", "").strip()
        example['tags'] = json.loads(clean_json)['discovered_techniques']
    except Exception:
        example['tags'] = []
    
    return example

def report_progress(url, value):
    """Reports evaluation progress to the backend."""
    with contextlib.suppress(BaseException):
        requests.post(f"{url}/training/progress", 
                      json={"stage": "evaluation", "value": value}, 
                      timeout=1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, required=True, help="Path to adapter")
    parser.add_argument("--base", type=str, required=True, help="Path to base model")
    parser.add_argument("--data", type=str, required=True, help="Path to test dataset (.jsonl)")
    parser.add_argument("--backend", type=str, default="http://localhost:8000", help="Backend URL")
    parser.add_argument("--output_dir", type=str, default="./model/benchmark_reports", help="Output directory for reports")
    parser.add_argument("--no-tqdm", action="store_true", help="Disable tqdm progress bar")
    args = parser.parse_args()

    # Load Model (supports direct adapter loading via Unsloth)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.adapter,
        max_seq_length = 2048,
        load_in_4bit = True,
        use_gradient_checkpointing = "unsloth",
    )
    FastLanguageModel.for_inference(model)
    
    # Load and Sample Dataset
    dataset = load_dataset("json", data_files=args.data, split="train") 
    dataset = dataset.shuffle(seed=42).select(range(min(7, len(dataset))))
    
    # Process Prompts and Run Inference
    dataset = dataset.map(lambda x: format_prompt(x, tokenizer))
    results = []
    
    iterator = dataset if args.no_tqdm else tqdm(dataset)
    for i, example in enumerate(iterator):
        inputs = tokenizer([example['prompt']], return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=512, use_cache=True, temperature=0.0)
            
        # Decode and evaluate
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        response_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        results.append(evaluate_response(response_text, example['tags']))
        
        report_progress(args.backend, int((i + 1) / len(dataset) * 100))
        
    # Aggregate Metrics
    total_docs = len(results)
    strict_success_count = sum(1 for r in results if r['parsing_status'] == 'Strict Success')
    non_empty_gold_docs = [r for r in results if r['has_gold_labels']]
    
    total_f1_doc_non_empty = sum(r['f1_doc'] for r in non_empty_gold_docs)
    exact_matches_count = sum(1 for r in results if r['exact_match'])
    
    parsing_success_rate = strict_success_count / total_docs if total_docs > 0 else 0
    mean_f1_doc_non_empty = total_f1_doc_non_empty / len(non_empty_gold_docs) if non_empty_gold_docs else 0
    exact_match_accuracy = exact_matches_count / total_docs if total_docs > 0 else 0
    
    # Print machine-readable scores for the Orchestrator
    print(f"FINAL_F1_SCORE: {mean_f1_doc_non_empty:.4f}")
    print(f"FINAL_EXACT_MATCH: {exact_match_accuracy:.4f}")
    
    # Write Final Report
    os.makedirs(args.output_dir, exist_ok=True)
    filename = f"benchmark_report_{int(time.time())}.txt"
    output_path = os.path.join(args.output_dir, filename)
    
    report = [
        "="*60,
        f"INFERENCE REPORT: {total_docs} documents",
        f"Adapter: {args.adapter}",
        f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "="*60,
        f"Parsing Success Rate (Strict JSON): {parsing_success_rate:.4f} ({strict_success_count}/{total_docs})",
        f"Exact-Match Accuracy: {exact_match_accuracy:.4f} ({exact_matches_count}/{total_docs})",
        f"Mean F1 (Non-empty gold docs): {mean_f1_doc_non_empty:.4f}" if non_empty_gold_docs else "Mean F1: N/A",
        "-"*60
    ]
    
    with open(output_path, "w", encoding="utf-8") as f: f.write("\n".join(report))
    print(f"Report written to: {output_path}")

if __name__ == "__main__":
    main()
