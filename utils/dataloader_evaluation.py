import json
from typing import List, Tuple

def load_qa_pairs_list(file_path: str) -> Tuple[List[str], List[str]]:
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

def load_qa_with_metadata(file_path: str):
    """
    Load questions, answers, and IDs from JSON file containing an array of QA pairs.
    Returns a list of dict (ids, questions, answers)
    """
    qa_dict = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            if 'id' in item and 'question' in item and 'answer' in item:
                # Handle list of answers
                if isinstance(item['answer'], list):
                    for i,ans in enumerate(item['answer']):
                        qa_dict.append({
                            "id": f"{item['id']}_{i}",
                            "question": item['question'],
                            "answer": ans
                        })
                # Handle single answer
                else:
                    qa_dict.append({
                        "id": f"{item['id']}_0",
                        "question": item['question'],
                        "answer": item['answer']
                    })      
    return qa_dict