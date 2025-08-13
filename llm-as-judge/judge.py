from Selene import Selene
import torch
import random
import os
import sys
from dotenv import load_dotenv
import json
import argparse
import re

# Add parent directory to Python path to find utils module
current_dir = os.path.dirname(os.path.abspath(__file__))  # llm-as-judge
parent_dir = os.path.dirname(current_dir)  # nos-rag-eval
sys.path.append(parent_dir)

from utils.dataloader_evaluation import load_questions_with_metadata
from prompts import (
    CONTEXT_RECALL_PROMPT,
    CONTEXT_PRECISION_PROMPT
)


env_path = os.path.join("/home/pablo.fernandez.rodriguez/configs/ragas.env")
load_dotenv(env_path)

def get_env_variable(var_name):
    """Retrieve an environment variable or raise an error if not set."""
    value = os.getenv(var_name)
    if not value:
        raise ValueError(f"{var_name} not set in .env file")
    return value

cache_dir = get_env_variable('CACHE_DIR')

def split_sentences(text): #Naive approach to split sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if len(s) > 3]

def build_context_recall_prompt(sentence, context):
    return CONTEXT_RECALL_PROMPT.format(
        sentence=sentence,
        context=context)

def build_context_precision_prompt(context_sentence, question, ground_truth):
    return CONTEXT_PRECISION_PROMPT.format(
        context_sentence=context_sentence,
        question=question,
        ground_truth=ground_truth
    )

def compute_context_recall(judge, contexts, ground_truth):
    gt_sentences = split_sentences(ground_truth)
    if not gt_sentences:
        return 0.0
    relevant_count = 0
    for sent in gt_sentences:
        for ctx in contexts:
            prompt = build_context_recall_prompt(sent, ctx)
            print(prompt)
            result = judge.evaluate(prompt)
            if "yes" in result.lower():
                relevant_count += 1
                break  # One supporting context is enough
    return relevant_count / len(gt_sentences)

def compute_context_recall_per_context(judge, contexts, ground_truth):
    gt_sentences = split_sentences(ground_truth)
    if not gt_sentences:
        return {ctx: 0.0 for ctx in contexts}

    context_scores = {}
    for ctx in contexts:
        supported_count = 0
        for sent in gt_sentences:
            prompt = build_context_recall_prompt(sent, ctx)
            result = judge.evaluate(prompt)
            if "yes" in result.lower():
                supported_count += 1
        context_scores[ctx] = supported_count / len(gt_sentences)
    return context_scores

def compute_context_precision(judge, contexts, question, ground_truth):
    all_sentences = []
    for ctx in contexts:
        all_sentences.extend(split_sentences(ctx))
    if not all_sentences:
        return 0.0
    relevant_count = 0
    for sent in all_sentences:
        prompt = build_context_precision_prompt(sent, question, ground_truth)
        result = judge.evaluate(prompt)
        #print(f"Evaluating context sentence: {sent}\nResult: {result}\n")
        if "yes" in result.lower():
            relevant_count += 1

    return relevant_count / len(all_sentences)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate with LLM-as-Judge Retrieval Results")
    parser.add_argument('--results', type=str, default=None, help='Path to results')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    judge_llm = Selene(cache_dir=cache_dir, device=device)
    with open(args.results) as f:
        eval_dataset = json.load(f)
    with open("/home/compartido/pabloF/nos-rag-eval/datasets/Revised_Dataset/preguntas_117_Revisado.json") as f:
        questions = json.load(f)
    # Select 5 random examples
    eval_dataset = eval_dataset[45:50]
    sample_size = 5
    for i, example in enumerate(eval_dataset):
        user_input = example['user_input']
        #assistant_response = example['response']
        reference_response = questions[i]['answer'][0]
        retrieved_contexts = [ context_json['context'] for context_json in example['retrieved_contexts']]
        print(f"--------------Evaluating question: {user_input}-----------------\n")
        #print(f"Assistant Response: {assistant_response}\n")
        #print(f"Reference Response: {reference_response}\n")


        # Evaluate context recall and precision
        recall = compute_context_recall(judge_llm, retrieved_contexts, reference_response)
        # separated_recall = compute_context_recall_per_context(judge_llm, retrieved_contexts, reference_response)
        #precision = compute_context_precision(judge_llm, retrieved_contexts, user_input, reference_response)
        # i=0
        # for _, score in separated_recall.items():
        #     print(f"Context {i}, Recall Score: {score:.2f}\n")
        #     i+=1
        # print(f"Total Context Recall: {recall:.2f}")
        print(f"Context Recall: {recall:.2f}\n")
        #print(f"Context Precision: {precision:.2f}\n")

