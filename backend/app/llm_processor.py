import re
import json

VALID_TAGS = {
    'REFERENCE_ERROR', 'WHATABOUTISM', 'STRAWMAN', 'EMOTIONAL_CONTENT', 
    'CHERRY_PICKING', 'FALSE_CAUSE', 'MISLEADING_CLICKBAIT', 'ANECDOTE', 
    'LEADING_QUESTIONS', 'EXAGGERATION', 'QUOTE_MINING'
}

def normalize_llm_response(content: str) -> dict:
    """
    Heals and normalizes LLM output by:
    1. Parsing JSON (or using regex if parsing fails).
    2. Mapping fuzzy keys (e.g., 'reason' -> 'reasoning').
    3. Validating and cleaning up technique tags (mapping typos).
    """
    # Try initial JSON parse
    parsed_content = None
    try:
        parsed_content = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        pass

    # --- PHASE 1: RESPONSE HEALING (Regex Recovery) ---
    if not isinstance(parsed_content, dict):
        # Extract reasoning
        reasoning_match = re.search(r'"reason\w*"\s*:\s*"(.*?)"', content, re.DOTALL)
        reasoning = reasoning_match.group(1) if reasoning_match else "Nie udało się wygenerować uzasadnienia."
        
        # Extract tags from [ "TAG", ... ]
        tags = []
        list_match = re.search(r'\[(.*?)\]', content, re.DOTALL)
        if list_match:
            raw_list = list_match.group(1)
            tags = [t.strip().strip('"\'') for t in raw_list.split(',') if t.strip()]
        
        parsed_content = {"reasoning": reasoning, "discovered_techniques": tags}
        print("DEBUG: Recovered content via regex heuristics")

    # --- PHASE 2: SCHEMA NORMALIZATION (Fuzzy Keys) ---
    if "reasoning" not in parsed_content:
        for k in parsed_content.keys():
            if k.startswith("reason"):
                parsed_content["reasoning"] = parsed_content[k]
                break
        if "reasoning" not in parsed_content:
             parsed_content["reasoning"] = "Brak uzasadnienia."

    if "discovered_techniques" not in parsed_content:
        for k in parsed_content.keys():
            if "technique" in k or "discovered" in k:
                 parsed_content["discovered_techniques"] = parsed_content[k]
                 break
        if "discovered_techniques" not in parsed_content:
            parsed_content["discovered_techniques"] = []

    # --- PHASE 3: TAG CLEANUP (Validation & Mapping) ---
    raw_tags = parsed_content.get("discovered_techniques", [])
    if not isinstance(raw_tags, list): raw_tags = []
        
    cleaned_tags = set()
    for tag in raw_tags:
        tag_upper = str(tag).upper().strip()
        if tag_upper in VALID_TAGS:
            cleaned_tags.add(tag_upper)
        else:
            if "EMOTIO" in tag_upper: cleaned_tags.add("EMOTIONAL_CONTENT")
            elif "CHERRY" in tag_upper: cleaned_tags.add("CHERRY_PICKING")
            elif "CLICKBAIT" in tag_upper: cleaned_tags.add("MISLEADING_CLICKBAIT")
            elif "QUOTE" in tag_upper: cleaned_tags.add("QUOTE_MINING")
            elif "ANECDOT" in tag_upper: cleaned_tags.add("ANECDOTE")
            else: cleaned_tags.add(tag_upper)
            
    parsed_content["discovered_techniques"] = list(cleaned_tags)
    return parsed_content
