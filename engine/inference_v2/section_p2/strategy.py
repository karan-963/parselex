import re
from typing import List

def post_process_predictions(
    tokens: List[dict], 
    predictions: List[str], 
    confidences: List[float] = None
) -> List[str]:
    """
    Tuning logic for Section Detection & Chunking.
    Applies keyword overrides only when classification confidence falls below 0.70.
    """
    final_predictions = list(predictions)
    if confidences is None:
        # If no confidences are provided, assume 0.0 (allow override)
        confidences = [0.0] * len(predictions)
        
    # Define keywords for section overrides
    rules = [
        (r'\b(education|university|college|degree|academic|coursework|gpa|school)\b', 'EDUCATION'),
        (r'\b(experience|employment|work history|professional background|career|internship)\b', 'EXPERIENCE'),
        (r'\b(skills|technologies|tools|languages|proficiencies|expertise)\b', 'SKILLS'),
        (r'\b(projects|github|personal projects|selected projects)\b', 'PROJECT'),
        (r'\b(summary|objective|profile|about me|career statement)\b', 'SUMMARY'),
        (r'\b(certification|certifications|awards|honors|achievements)\b', 'OTHER')
    ]
    
    # Process each prediction
    for idx, (pred, conf) in enumerate(zip(predictions, confidences)):
        # Restrict keyword overrides to execute only when the model's confidence is below 0.70
        if conf < 0.70:
            tok = tokens[idx]
            tok_text = tok.get("token", "").lower()
            
            for pattern, target_section in rules:
                if re.search(pattern, tok_text):
                    final_predictions[idx] = target_section
                    break  # Apply the first matching rule
                    
    return final_predictions
