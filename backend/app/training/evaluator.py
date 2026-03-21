import json
import re
from sklearn.metrics import f1_score
from typing import List, Dict

class AutoBenchmarker:
    """Utility class for automated evaluation of LLM response parsing and classification accuracy."""
    def __init__(self, technique_mapping: Dict):
        self.technique_mapping = technique_mapping

    def evaluate_response(self, response_text: str, ground_truth_tags: List[str]) -> Dict:
        """
        Evaluates a single LLM response, calculating Parsing Success Rate (PSR) and F1 Score.
        """
        parsed_tags = []
        parsing_status = 'Failed'

        # Pre-processing: Clean markdown wrappers
        clean_text = response_text.replace("```json", "").replace("```", "").strip()
        
        # Phase 1: Structured JSON Parsing (PSR)
        try:
            parsed_output = json.loads(clean_text)
            if isinstance(parsed_output, dict):
                parsed_tags = parsed_output.get("discovered_techniques", [])
                parsing_status = 'Strict Success'
        except (json.JSONDecodeError, ValueError):
            # Phase 2: Format Correction (FCR via RegEx soft parsing)
            match = re.search(r'\[(.*?)\]', clean_text, re.DOTALL)
            if match:
                try:
                    candidate = json.loads(f"[{match.group(1)}]")
                    if isinstance(candidate, list):
                        parsed_tags = candidate
                        parsing_status = 'Recovered'
                except: pass

        # Phase 3: Classification Metric (Macro F1)
        f1 = self.calculate_f1(parsed_tags, ground_truth_tags)
        
        return {
            "parsing_status": parsing_status,
            "f1_score": f1,
            "parsed_tags": parsed_tags,
            "ground_truth": ground_truth_tags
        }

    def calculate_f1(self, predicted: List[str], actual: List[str]) -> float:
        """Calculates Macro F1 score between predicted and actual tags."""
        predicted = [str(t) for t in predicted]
        actual = [str(t) for t in actual]
        
        all_classes = list(self.technique_mapping.keys())
        y_true = [1 if cls in actual else 0 for cls in all_classes]
        y_pred = [1 if cls in predicted else 0 for cls in all_classes]
        
        # Perfect match for empty-to-empty
        if sum(y_true) == 0 and sum(y_pred) == 0:
            return 1.0
        
        return f1_score(y_true, y_pred, average='macro', zero_division=0)
