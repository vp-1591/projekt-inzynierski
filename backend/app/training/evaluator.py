import json
import re
from sklearn.metrics import f1_score
from typing import List, Dict

class AutoBenchmarker:
    def __init__(self, technique_mapping: Dict):
        self.technique_mapping = technique_mapping

    def evaluate_response(self, response_text: str, ground_truth_tags: List[str]) -> Dict:
        """
        Implementation of 2.4: PSR, FCR, and Classification Performance
        """
        parsed_tags = []
        parsing_status = 'Failed' # Failed, Strict Success, Recovered (FCR)

        # 1. Parsing Success Rate (PSR) - Section 2.4.Metric 1
        clean_text = response_text.replace("```json", "").replace("```", "").strip()
        try:
            parsed_output = json.loads(clean_text)
            if isinstance(parsed_output, dict):
                parsed_tags = parsed_output.get("discovered_techniques", [])
                parsing_status = 'Strict Success'
        except (json.JSONDecodeError, ValueError):
            # 2. Format Correction Rate (FCR) - Section 2.4.Metric 2 (RegEx soft parsing)
            match = re.search(r'\[(.*?)\]', clean_text, re.DOTALL)
            if match:
                extracted_content = f"[{match.group(1)}]"
                try:
                    parsed_tags = json.loads(extracted_content)
                    if isinstance(parsed_tags, list):
                        parsing_status = 'Recovered'
                except:
                    pass

        # 3. Classification Performance - Section 2.4.Metric 3 (Macro F1)
        f1 = self.calculate_f1(parsed_tags, ground_truth_tags)
        
        return {
            "parsing_status": parsing_status,
            "f1_score": f1,
            "parsed_tags": parsed_tags,
            "ground_truth": ground_truth_tags
        }

    def calculate_f1(self, predicted: List[str], actual: List[str]) -> float:
        predicted = [str(t) for t in predicted]
        actual = [str(t) for t in actual]
        
        all_classes = list(self.technique_mapping.keys())
        y_true = [1 if cls in actual else 0 for cls in all_classes]
        y_pred = [1 if cls in predicted else 0 for cls in all_classes]
        
        if sum(y_true) == 0 and sum(y_pred) == 0:
            return 1.0
        
        return f1_score(y_true, y_pred, average='macro', zero_division=0)
