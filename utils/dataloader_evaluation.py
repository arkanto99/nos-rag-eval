import json
from typing import List, Tuple

def load_qa_pairs(file_path: str) -> Tuple[List[str], List[str]]:
    """
    Load questions and answers from JSON file containing an array of QA pairs.
    Returns tuple of (questions, answers) lists.
    """
    questions = []
    answers = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            if 'question' in item and 'answer' in item:
                # Handle list of answers
                if isinstance(item['answer'], list):
                    for ans in item['answer']:
                        questions.append(item['question'])
                        answers.append(ans)
                # Handle single answer
                else:
                    questions.append(item['question'])
                    answers.append(item['answer'])
                    
    return questions, answers