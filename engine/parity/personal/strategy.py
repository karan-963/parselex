import re
from typing import List

_PHONE_STRIP = re.compile(r'[\s\-\(\)\.\+]')
_PHONE_FULL  = re.compile(r'^\+?[\d][\d\s\-\.\(\)]{5,14}[\d]$')

def _is_phone_token(text: str) -> bool:
    """Return True if text looks like a phone number."""
    digits_only = _PHONE_STRIP.sub('', text)
    if digits_only.isdigit() and 7 <= len(digits_only) <= 15:
        return True
    if _PHONE_FULL.match(text.strip()):
        stripped = _PHONE_STRIP.sub('', text)
        if stripped.lstrip('+').isdigit() and 7 <= len(stripped.lstrip('+')) <= 15:
            return True
    return False

def post_process_predictions(tokens: List[dict], predictions: List[str]) -> List[str]:
    """
    Apply standalone layout, keyword, and visual text heuristic rules.
    """
    final_predictions = list(predictions)
    PROTECTED_LABELS = {"B-NAME", "I-NAME", "B-LOCATION", "I-LOCATION", "B-POSITION", "I-POSITION"}
    email_regex = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
    
    # 1. Group tokens by (page, lineIndex) to analyze line structure
    line_groups = {}
    for idx, t in enumerate(tokens):
        key = (t.get("page", 0), t.get("lineIndex", 0))
        if key not in line_groups:
            line_groups[key] = []
        line_groups[key].append((idx, t))
        
    project_verbs = {"designed", "developed", "supported", "built", "implemented", "created", "managed", "led", "assisted"}
    tech_keywords = {"python", "java", "sql", "devops", "cloud", "c++", "javascript", "typescript", "react", "aws", "docker", "git", "kubernetes", "c#", "html", "css"}
    personal_keywords = {"hobbies", "dob", "languages", "hometown", "nationality", "interests"}
    
    # 2. Apply line-level visual and layout safeguards (experience/tech leakage prevention)
    for key, item_list in line_groups.items():
        # Sort tokens in reading order along the line
        item_list.sort(key=lambda x: x[1].get("x0", 0.0))
        
        line_tokens_text = [x[1].get("token", "") for x in item_list]
        line_str = " ".join(line_tokens_text)
        line_str_lower = line_str.lower().strip()
        
        # Clean leading bullet points/dashes
        clean_line_str = re.sub(r'^[•▪\-*o■–—·✔➢\uf0a7\uf0d8\u2022\u2023\u2043\u254b\u25b8\s]+', '', line_str_lower)
        
        # Rule A: Line starts with project/experience verb -> Override the experience part
        starts_with_verb = False
        for verb in project_verbs:
            if clean_line_str.startswith(verb):
                starts_with_verb = True
                break
                
        # Check if line contains a personal keyword and where it starts
        has_personal_kw = False
        personal_kw_pos = -1
        for kw in personal_keywords:
            pos = clean_line_str.find(kw)
            if pos != -1:
                has_personal_kw = True
                personal_kw_pos = pos
                break
                
        # Find if there is a separator token
        sep_idx = -1
        for i, (idx, t) in enumerate(item_list):
            if t.get("token", "").strip() in {"|", "•", "·", "▪", "■", "–", "—"}:
                sep_idx = i
                break
                
        if starts_with_verb:
            if sep_idx != -1:
                # Stitched line: verb is on the left, personal info might be on the right
                # Override tokens to the left of the separator
                for idx, _ in item_list[:sep_idx]:
                    final_predictions[idx] = "O"
            else:
                # Not stitched, override the whole line
                for idx, _ in item_list:
                    final_predictions[idx] = "O"
            continue
            
        # Rule B: Spoken languages do not contain tech keywords
        # Only run this if the line actually mentions languages, hobbies, or interests
        is_lang_or_hobby_line = any(kw in clean_line_str for kw in {"language", "languages", "hobbies", "hobby", "interests"})
        if is_lang_or_hobby_line:
            for idx, t in item_list:
                t_text = t.get("token", "").lower().strip()
                if t_text in tech_keywords:
                    final_predictions[idx] = "O"
                    
        # Rule C: Column stitched personal keyword starts late in the line (>= 15 characters)
        if has_personal_kw and personal_kw_pos >= 15:
            if sep_idx != -1:
                # Override tokens to the left of the separator
                for idx, _ in item_list[:sep_idx]:
                    final_predictions[idx] = "O"
            else:
                # Find the token index where the personal keyword starts
                kw_token_i = -1
                for i, (idx, t) in enumerate(item_list):
                    t_text = t.get("token", "").lower().strip()
                    if any(kw in t_text for kw in personal_keywords):
                        kw_token_i = i
                        break
                if kw_token_i != -1:
                    # Override tokens to the left of the personal keyword
                    for idx, _ in item_list[:kw_token_i]:
                        final_predictions[idx] = "O"
            continue
            
        # If personal keyword starts early (< 15 characters) but the line is stitched with experience/skills
        if has_personal_kw and personal_kw_pos < 15 and sep_idx != -1:
            right_part = clean_line_str[personal_kw_pos:]
            has_right_tech_or_verb = any(tech in right_part for tech in tech_keywords) or any(verb in right_part for verb in project_verbs)
            if has_right_tech_or_verb:
                for idx, _ in item_list[sep_idx:]:
                    final_predictions[idx] = "O"
            continue
            
    # 3. Apply token-level standard heuristic overrides (based strictly on visual token text)
    for idx, t in enumerate(tokens):
        text = t.get("token", "").strip()
        text_low = text.lower()
        
        # Rule 0: Pure Layout Punctuation -> O
        if text in {",", "|", "(", ")", "[", "]", "#", "§", "-", "•", "·", "/"}:
            final_predictions[idx] = "O"
            continue
            
        # Rule 1: Structural Anchor Protections (plain-text keywords are never entities)
        if text_low in ["email", "e-mail", "mail", "mobile", "phone", "tel", "contact", "linkedin", "github"]:
            final_predictions[idx] = "O"
            continue
            
        # Rule 2: Absolute Email Verification
        # Only override if the model's raw prediction is not already an email tag
        if final_predictions[idx] not in ("B-EMAIL", "I-EMAIL"):
            if bool(email_regex.search(text_low)):
                prev_text = tokens[idx-1].get("token", "").lower() if idx > 0 else ""
                if idx == 0 or not email_regex.search(prev_text):
                    final_predictions[idx] = "B-EMAIL"
                else:
                    final_predictions[idx] = "I-EMAIL"
                continue
            
        # Rule 3: LinkedIn Override (strictly check for domain name in text)
        if final_predictions[idx] not in ("B-LINKEDIN", "I-LINKEDIN"):
            if "linkedin.com" in text_low:
                prev_text = tokens[idx-1].get("token", "").lower() if idx > 0 else ""
                if idx == 0 or "linkedin.com" not in prev_text:
                    final_predictions[idx] = "B-LINKEDIN"
                else:
                    final_predictions[idx] = "I-LINKEDIN"
                continue
            
        # Rule 4: GitHub Override (strictly check for domain name in text)
        if final_predictions[idx] != "B-GITHUB":
            if "github.com" in text_low:
                final_predictions[idx] = "B-GITHUB"
                continue
            
        # Rule 5: Phone overrides
        if final_predictions[idx] not in ("B-PHONE", "I-PHONE") and _is_phone_token(text):
            prev_is_phone = _is_phone_token(tokens[idx-1].get("token", "")) if idx > 0 else False
            if not prev_is_phone:
                final_predictions[idx] = "B-PHONE"
            else:
                final_predictions[idx] = "I-PHONE"
            continue
            
    return final_predictions


def sanitize_bio_sequence(pred_labels: List[str]) -> List[str]:
    """
    Sanitize sequence predictions to enforce BIO consistency:
    1. If a segment is an I- tag but not preceded by B- or I- of the same entity type, coerce it to B-.
    2. If a segment is a B- tag but is preceded by B- or I- of the same entity type, coerce it to I-.
    """
    sanitized = []
    prev_entity = None
    
    for lbl in pred_labels:
        if lbl == "O":
            sanitized.append("O")
            prev_entity = None
            continue
            
        if "-" in lbl:
            prefix, entity = lbl.split("-", 1)
        else:
            sanitized.append(lbl)
            prev_entity = None
            continue
            
        if prefix == "I":
            if prev_entity != entity:
                sanitized.append(f"B-{entity}")
                prev_entity = entity
            else:
                sanitized.append(lbl)
        elif prefix == "B":
            if prev_entity == entity:
                sanitized.append(f"I-{entity}")
            else:
                sanitized.append(lbl)
                prev_entity = entity
        else:
            sanitized.append(lbl)
            prev_entity = None
            
    return sanitized
